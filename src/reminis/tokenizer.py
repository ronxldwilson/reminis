"""Turning text into token ids using only what the database holds.

The vocabulary, the merges, the scores and the pre-tokenizer's name are
all rows in ``model_meta``, so nothing here loads a tokenizer file or
reaches for the ``tokenizers`` package. Three families cover what the
converter has met:

  * ``BPETokenizer`` -- byte-level BPE over GPT-2's reversible byte map,
    which is llama-3, qwen2, smollm and gpt-2
  * ``PieceBPETokenizer`` -- the same merges written as sentencepiece
    pieces, where a leading space is a lower-one-eighth block
  * ``SPMTokenizer`` -- sentencepiece proper, scored rather than merged,
    resolved with a heap because the longest match is not the best one

Which one a database wants is decided by ``build_tokenizer``, from the
model's own metadata rather than from its name. Getting this wrong is
quiet -- the text still decodes, one token off -- so the suites check it
against llama.cpp token for token rather than by eye.
"""

import heapq
import re
from functools import lru_cache

from reminis.errors import UnsupportedModel
from reminis.meta import _int_or_none, _parse_array

# ---------------------------------------------------------------- tokenizer


@lru_cache(maxsize=1)
def _byte_unicode_maps():
    """GPT-2's reversible byte-to-printable-character mapping.

    Byte-level BPE needs every one of the 256 byte values to be a character
    it can merge, and control characters and spaces are not usable, so GPT-2
    maps the unprintable ones into an unused Unicode range. This is why a
    space shows up as 'Ġ' in the stored vocabulary.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_to_char = {b: chr(b) for b in printable}
    spare = 0
    for b in range(256):
        if b not in byte_to_char:
            byte_to_char[b] = chr(256 + spare)
            spare += 1
    return byte_to_char, {c: b for b, c in byte_to_char.items()}


# GPT-2's pre-tokenizer splits text before BPE ever runs, and the split rules
# are part of the tokenizer: change them and the ids change. The originals are
# written with \p{L} and \p{N}, which Python's `re` does not have.
#
# `\w` is the closest thing available, and it is off by exactly one character:
# it counts underscore as a word character where \p{L} and \p{N} do not. That
# one character matters. Left uncorrected, an underscore matches neither the
# letter class nor the punctuation class, the pre-tokenizer skips it, and
# `and_underscores` silently loses a token relative to every other
# implementation. So the classes below are built to put it back:
#
#   _LETTER  \p{L}                       word characters that are not digits
#                                        or underscore
#   _OTHER   [^\s\p{L}\p{N}]             not space, letter or digit -- which
#                                        means underscore is explicitly in
_LETTER = r"[^\W\d_]"
_OTHER = r"(?:[^\s\w]|_)"
_PRETOKENIZERS = {
    # llama 3 and qwen 2 share this shape: digits in runs of at most three,
    # and a leading non-letter allowed to attach to a word.
    "llama-bpe": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        rf"|(?:[^\r\n\w]|_)?{_LETTER}+"
        r"|\d{1,3}"
        rf"| ?{_OTHER}+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    ),
    # The GPT-2 original, which SmolLM and most older BPE vocabularies use.
    "default": (
        r"'s|'t|'re|'ve|'m|'ll|'d"
        rf"| ?{_LETTER}+"
        r"| ?\d+"
        rf"| ?{_OTHER}+"
        r"|\s+(?!\S)"
        r"|\s+"
    ),
}
# gpt-4o's splitter needs Unicode case categories -- an uppercase run
# followed by a lowercase run, with a contraction attached to the word
# rather than split from it, so "It's" is one piece and not two. Python's
# `re` has no \p{Lu}, so this one is only available when the `regex`
# module is installed, and the tokenizer says so rather than quietly
# splitting differently from every other implementation.
_GPT4O_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# qwen 3.5's splitter, which is not qwen2's and not llama 3's. Two
# differences, and the first one matters on every number in every prompt:
#
#   digits    `\p{N}` matches one digit at a time, where llama 3 takes runs
#             of up to three. "2026" is four pieces here and two there, and
#             a model asked to do arithmetic on the wrong pieces answers
#             fluently and wrongly.
#   marks     `\p{M}` joins combining marks to the letter they modify, so a
#             decomposed accent stays with its base rather than splitting
#             off into the punctuation class.
#
# Like gpt-4o's, this needs Unicode general categories that Python's `re`
# does not have, so it is only available with the `regex` module and says
# so rather than silently splitting a different way.
_QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

_PRETOKENIZERS["qwen2"] = _PRETOKENIZERS["llama-bpe"]
_PRETOKENIZERS["smollm"] = _PRETOKENIZERS["default"]
_PRETOKENIZERS["gpt-2"] = _PRETOKENIZERS["default"]


def build_tokenizer(meta: dict):
    """The tokenizer this model was trained with, rebuilt from the database.

    Two families cover everything reminis runs. ``gpt2`` is byte-level BPE,
    driven by an ordered merge list. ``llama`` is SentencePiece, which has no
    merge list at all -- it merges by a score attached to each token, so the
    two share almost no machinery despite both being called BPE.
    """
    model = meta.get("tokenizer.ggml.model")
    if model == "gpt2":
        return BPETokenizer(meta)
    if model in ("llama", "spm"):
        # SentencePiece proper merges by score. Some conversions keep the
        # SentencePiece alphabet but carry a merge list instead, and fill
        # every score with the same placeholder -- which is a merge-ranked
        # BPE wearing a SentencePiece vocabulary, and has to be encoded as
        # one. A single distinct score means the scores decide nothing.
        if _scores_are_inert(meta) and _parse_array(meta, "tokenizer.ggml.merges"):
            return PieceBPETokenizer(meta)
        return SPMTokenizer(meta)
    if model in ("gemma4", "gemma3", "gemma2", "gemma"):
        return PieceBPETokenizer(meta)
    raise UnsupportedModel(
        f"This model's tokenizer is '{model or 'missing'}'. reminis run "
        f"implements byte-level BPE ('gpt2'), SentencePiece ('llama') and "
        f"merge-ranked SentencePiece ('gemma')."
    )


def _scores_are_inert(meta: dict) -> bool:
    """Whether the token scores carry no information.

    A conversion that drops SentencePiece's scores writes the same
    placeholder for every token. Merging by them would then be merging by
    nothing, in whatever order the vocabulary happens to be in.
    """
    try:
        scores = _numeric_array(meta, "tokenizer.ggml.scores")
    except Exception:
        return False
    return bool(scores) and len(set(scores)) == 1


def _numeric_array(meta: dict, key: str) -> list:
    """A GGUF numeric array, which arrives as one-element lists per entry."""
    return [
        v[0] if isinstance(v, (list, tuple)) else v
        for v in _parse_array(meta, key)
    ]


class _SpecialTokens:
    """The part of encoding that does not depend on which BPE this is.

    All three tokenizers below open the same way: emit the beginning-of-text
    token, cut the text on the special-token pattern, hand back a special's
    id directly rather than letting the merge algorithm take it apart, and
    run everything else through their own encoder. That was written out
    three times, which is three places to fix when the handling of a special
    is wrong -- and getting one of them wrong is silent, because a special
    that goes through the merge path still produces plausible ids.

    Subclasses supply `_encode_chunk`, which sees only ordinary text.
    """

    def _init_specials(self, types) -> None:
        """Collect the special tokens and the pattern that finds them.

        Longest first, so a vocabulary holding both `<|im_end|>` and `<|im|>`
        cuts at the longer one -- `re` alternation takes the first branch
        that matches, not the longest.
        """
        self.specials = sorted(
            (self.tokens[i] for i, t in enumerate(types) if int(t) in (3, 4)),
            key=len, reverse=True,
        )
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(s) for s in self.specials) + ")")
            if self.specials else None
        )

    def _encode_chunk(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = []
        if add_special and self.add_bos and self.bos_id is not None:
            ids.append(self.bos_id)

        chunks = (
            self._special_pattern.split(text) if self._special_pattern else [text]
        )
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.ids and chunk in self.specials:
                ids.append(self.ids[chunk])
                continue
            ids.extend(self._encode_chunk(chunk))
        return ids


class SPMTokenizer(_SpecialTokens):
    """SentencePiece, as llama.cpp implements it.

    There is no merge list. Every token carries a score, and the algorithm
    repeatedly merges whichever adjacent pair forms the highest-scoring token
    in the vocabulary -- so the vocabulary itself encodes the merge order.
    Text is pre-escaped by replacing spaces with U+2581, which is why a
    leading space appears in almost every token.

    Anything with no token at all falls back to one token per *byte*, which
    is why the vocabulary contains 256 entries named `<0x00>` through
    `<0xFF>`.
    """

    SPACE = "\u2581"

    def __init__(self, meta: dict):
        self.tokens = _parse_array(meta, "tokenizer.ggml.tokens")
        self.scores = _numeric_array(meta, "tokenizer.ggml.scores")
        types = _numeric_array(meta, "tokenizer.ggml.token_type")
        if not self.tokens or not self.scores:
            raise UnsupportedModel(
                "The database has no SentencePiece vocabulary in it."
            )

        self.ids = {t: i for i, t in enumerate(self.tokens)}
        # Type 6 is BYTE: the fallback for text with no token of its own.
        self.byte_ids = {}
        for i, t in enumerate(types):
            if int(t) == 6:
                text = self.tokens[i]
                if len(text) == 6 and text.startswith("<0x"):
                    self.byte_ids[int(text[3:5], 16)] = i
        self.id_to_byte = {i: b for b, i in self.byte_ids.items()}

        self._init_specials(types)

        self.bos_id = _int_or_none(meta.get("tokenizer.ggml.bos_token_id"))
        self.eos_id = _int_or_none(meta.get("tokenizer.ggml.eos_token_id"))
        # SentencePiece models add the beginning-of-text token by default,
        # and prepend a space so the first word looks like any other.
        self.add_bos = str(
            meta.get("tokenizer.ggml.add_bos_token", "True")
        ).lower() == "true"
        self.add_space_prefix = str(
            meta.get("tokenizer.ggml.add_space_prefix", "True")
        ).lower() == "true"
        self.chat_template = meta.get("tokenizer.chat_template", "")

    def _encode_chunk(self, text: str) -> list[int]:
        # Every stretch of ordinary text gets the leading space, not just the
        # first: "x[INST]y" encodes its "y" as "_y", the same as its "x".
        # Checked against llama.cpp, which does the same.
        if self.add_space_prefix:
            text = " " + text
        text = text.replace(" ", self.SPACE)
        if not text:
            return []

        chars = list(text)
        n = len(chars)
        # A doubly linked list over the characters, so merging is a pointer
        # update rather than a rebuild of the sequence.
        length = [1] * n
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[-1] = -1

        heap = []

        def consider(left, right):
            if left == -1 or right == -1:
                return
            piece = "".join(chars[left:left + length[left] + length[right]])
            token = self.ids.get(piece)
            if token is None:
                return
            # A min-heap over (-score, left) is a max-heap over score that
            # breaks ties toward the earlier position, which is the order
            # llama.cpp's priority queue produces.
            heapq.heappush(heap, (-self.scores[token], left, right, len(piece)))

        for i in range(1, n):
            consider(i - 1, i)

        while heap:
            _, left, right, size = heapq.heappop(heap)
            if length[left] == 0 or length[right] == 0:
                continue
            if length[left] + length[right] != size:
                continue

            length[left] += length[right]
            length[right] = 0
            nxt[left] = nxt[right]
            if nxt[right] != -1:
                prev[nxt[right]] = left

            consider(prev[left], left)
            consider(left, nxt[left])

        out = []
        i = 0
        while i != -1:
            self._emit("".join(chars[i:i + length[i]]), out)
            i = nxt[i]
        return out

    def _emit(self, piece, out):
        """One surviving run to ids, or to its raw bytes if it has no token.

        A run only ever grew by merging into a string the vocabulary
        contains, so anything left without a token is a single character the
        model has never seen -- and those become one token per UTF-8 byte.
        """
        token = self.ids.get(piece)
        if token is not None:
            out.append(token)
            return
        for byte in piece.encode("utf-8"):
            if byte in self.byte_ids:
                out.append(self.byte_ids[byte])

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for i in ids:
            if i in self.id_to_byte:
                out.append(self.id_to_byte[i])
                continue
            if 0 <= i < len(self.tokens):
                out += self.tokens[i].replace(self.SPACE, " ").encode("utf-8")
        text = out.decode("utf-8", errors="replace")
        return text[1:] if self.add_space_prefix and text.startswith(" ") else text

    def decode_one(self, token_id: int) -> str:
        if token_id in self.id_to_byte:
            return bytes([self.id_to_byte[token_id]]).decode("utf-8", errors="replace")
        if 0 <= token_id < len(self.tokens):
            return self.tokens[token_id].replace(self.SPACE, " ")
        return ""


class BPETokenizer(_SpecialTokens):
    """Byte-level BPE, rebuilt from the vocabulary and merges in the database."""

    def __init__(self, meta: dict):
        self.tokens = _parse_array(meta, "tokenizer.ggml.tokens")
        merges = _parse_array(meta, "tokenizer.ggml.merges")
        if not self.tokens or not merges:
            raise UnsupportedModel(
                "The database has no tokenizer vocabulary in it, so there is "
                "nothing to encode with."
            )

        self.ids = {t: i for i, t in enumerate(self.tokens)}
        self.ranks = {tuple(m.split(" ")): i for i, m in enumerate(merges)}

        # Token types 3 and 4 are CONTROL and USER_DEFINED -- <|im_start|>
        # and friends, and markers like <think> that a chat template writes
        # as literal text. Both must be matched whole rather than run
        # through the splitter, which would cut <think> at the boundary
        # between its punctuation and its letters and leave BPE unable to
        # rebuild it: the pieces are all real tokens, so nothing complains,
        # and the model is handed a sequence it was never trained on.
        #
        # Only CONTROL was collected here, while the SentencePiece
        # tokenizer above already took both. That difference was the bug
        # rather than a distinction being drawn.
        #
        # Numeric GGUF arrays come back as one-element lists per entry,
        # since that is how the reader hands them over.
        types = [
            t[0] if isinstance(t, (list, tuple)) else t
            for t in _parse_array(meta, "tokenizer.ggml.token_type")
        ]
        self._init_specials(types)

        pre = meta.get("tokenizer.ggml.pre", "default")
        if pre in ("gpt-4o", "qwen35"):
            unicode_pattern = _GPT4O_PATTERN if pre == "gpt-4o" else _QWEN35_PATTERN
            try:
                import regex
            except ImportError:
                raise UnsupportedModel(
                    f"This model uses the {pre} pre-tokenizer, which needs "
                    f"Unicode general categories that Python's `re` does not "
                    f"have.\n"
                    f"  pip install regex"
                )
            self.pattern = regex.compile(unicode_pattern)
        else:
            self.pattern = re.compile(
                _PRETOKENIZERS.get(pre, _PRETOKENIZERS["default"])
            )

        self.bos_id = _int_or_none(meta.get("tokenizer.ggml.bos_token_id"))
        self.eos_id = _int_or_none(meta.get("tokenizer.ggml.eos_token_id"))
        self.add_bos = str(meta.get("tokenizer.ggml.add_bos_token", "False")).lower() == "true"
        self.chat_template = meta.get("tokenizer.chat_template", "")

        self._byte_to_char, self._char_to_byte = _byte_unicode_maps()
        self._bpe_cache: dict[str, list[str]] = {}

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = []
        for piece in self.pattern.findall(chunk):
            encoded = "".join(self._byte_to_char[b] for b in piece.encode("utf-8"))
            for token in self._bpe(encoded):
                token_id = self.ids.get(token)
                if token_id is None:
                    # Every single byte is in the vocabulary, so falling
                    # back to bytes always terminates.
                    ids.extend(self.ids[c] for c in token if c in self.ids)
                else:
                    ids.append(token_id)
        return ids

    def _bpe(self, word: str) -> list[str]:
        cached = self._bpe_cache.get(word)
        if cached is not None:
            return cached

        parts = list(word)
        while len(parts) > 1:
            pairs = zip(parts, parts[1:])
            best, best_rank = None, None
            for i, pair in enumerate(pairs):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = i, rank
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]

        self._bpe_cache[word] = parts
        return parts

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.tokens[i] for i in ids if 0 <= i < len(self.tokens))
        raw = bytes(self._char_to_byte.get(c, ord("?") if ord(c) > 255 else ord(c))
                    for c in text)
        return raw.decode("utf-8", errors="replace")

    def decode_one(self, token_id: int) -> str:
        return self.decode([token_id])


class PieceBPETokenizer(BPETokenizer):
    """BPE over a SentencePiece vocabulary, as Gemma's is stored.

    It is BPE -- an ordered merge list decides everything -- but the
    alphabet is SentencePiece's rather than GPT-2's, and the difference is
    not cosmetic:

      * a space is the character `▁`, not GPT-2's `Ġ`, and the merges are
        written in terms of it;
      * a byte with no piece of its own appears as the literal text
        `<0xE2>` rather than as a character in a remapped Unicode range.

    So the byte-level mapping the parent class applies is exactly wrong
    here, and both directions are overridden. What is inherited is the part
    that is genuinely shared: merge ranks, the merge loop, and the handling
    of control tokens.
    """

    SPACE = "▁"

    def __init__(self, meta: dict):
        super().__init__(meta)
        # A vocabulary this size is worth one pass to find the byte pieces,
        # rather than formatting a string per byte on every fallback.
        self._byte_piece = {}
        for value in range(256):
            piece = f"<0x{value:02X}>"
            if piece in self.ids:
                self._byte_piece[value] = piece
        self._piece_byte = {p: v for v, p in self._byte_piece.items()}
        # SentencePiece models differ on whether an implicit space is added
        # in front of the text; Gemma's says not to.
        self.add_space_prefix = str(
            meta.get("tokenizer.ggml.add_space_prefix", "True")
        ).lower() == "true"

    def _split(self, text: str) -> list[str]:
        """Words, each carrying its own leading space.

        Merging the whole text at once would be quadratic in its length and
        would also let a merge straddle a space that no training example
        ever presented as one piece. Splitting before each space marker is
        what SentencePiece itself does.
        """
        parts, current = [], ""
        for char in text:
            if char == self.SPACE and current:
                parts.append(current)
                current = char
            else:
                current += char
        if current:
            parts.append(current)
        return parts

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = []
        marked = chunk.replace(" ", self.SPACE)
        if self.add_space_prefix and not marked.startswith(self.SPACE):
            marked = self.SPACE + marked
        for word in self._split(marked):
            for token in self._bpe(word):
                token_id = self.ids.get(token)
                if token_id is not None:
                    ids.append(token_id)
                    continue
                # No piece for this text: spell it out a byte at a time,
                # which the vocabulary always has room for.
                for value in token.encode("utf-8"):
                    piece = self._byte_piece.get(value)
                    if piece is not None:
                        ids.append(self.ids[piece])
        return ids

    def decode(self, ids: list[int]) -> str:
        # Byte pieces have to be gathered into runs before decoding, since a
        # multi-byte character is spelled as several of them and neither
        # half is valid UTF-8 alone.
        out, pending = [], bytearray()

        def flush():
            if pending:
                out.append(pending.decode("utf-8", errors="replace"))
                pending.clear()

        for i in ids:
            if not 0 <= i < len(self.tokens):
                continue
            piece = self.tokens[i]
            value = self._piece_byte.get(piece)
            if value is not None:
                pending.append(value)
                continue
            flush()
            out.append(piece.replace(self.SPACE, " "))
        flush()
        return "".join(out)
