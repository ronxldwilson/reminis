"""Run a model straight out of its database.

Everything a forward pass needs is already in the file: the weights are rows
in ``tensors``, the hyperparameters and the tokenizer's vocabulary and merges
are rows in ``model_meta``. So this module loads no config, downloads nothing,
and imports neither torch nor llama.cpp -- it is numpy over the result of a
``SELECT``.

The point is not speed. It is that a reminis database is a *complete*,
self-contained model rather than an archive of one: if it can generate text,
then the conversion kept everything that mattered, and a merged or
rolled-back or delta-applied database can be checked by asking it to speak
rather than by comparing hashes.

Speed, measured rather than assumed, turns out to be less embarrassing than
expected -- roughly 0.7-0.9x llama.cpp's CPU token generation on the same
F16 models, since both end up in the platform BLAS for the large matrix
multiplies. Against llama.cpp on the GPU it is 2.5-3x slower, and against a
quantized model it does not compete at all, because it cannot run one.

``--stream`` takes that further. In streaming mode no weight is ever cached:
every matrix multiplication in every layer re-reads its operand from SQLite
and throws it away, so peak memory is one layer rather than one model. It is
slow, and it demonstrates the thing the whole project is about -- that the
model is data in a database, paged in on demand, not a file that must fit in
RAM.

Scope, deliberately narrow and loudly enforced:

  * llama-family and qwen2 architectures from GGUF (rotary, RMSNorm, SwiGLU,
    grouped-query attention), which covers llama, qwen2, smollm, mistral
  * float weights -- F32, F16, BF16. Quantized blocks are not decoded here
  * a byte-level BPE tokenizer stored in the database

Anything else raises rather than approximates, because a forward pass that
guesses produces fluent-looking nonsense, which is worse than an error.
"""

import ast
import math
import sqlite3
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from reminis.dtypes import is_float_dtype, to_float32

# Architectures whose block structure this file actually implements. The
# value is the rotary layout llama.cpp uses for that architecture, and it is
# not cosmetic: applying the wrong one produces confident gibberish.
#
#   "norm" rotates adjacent pairs (0,1), (2,3), ...   -- llama, mistral
#   "neox" rotates halves,  i with i + head_dim/2     -- qwen2
SUPPORTED_ARCHS = {
    "llama": "norm",
    "mistral": "norm",
    "qwen2": "neox",
    "qwen2moe": "neox",
}


class UnsupportedModel(Exception):
    """The database holds a model this forward pass does not implement."""


# ---------------------------------------------------------------- weights


class WeightStore:
    """Tensor access backed by SQLite.

    Cached mode reads each tensor once and keeps the float32 copy. Streaming
    mode reads it every single time it is used and keeps nothing, which is
    the mode that makes the "the model lives in the database" claim literal.
    """

    def __init__(self, db_path: str, stream: bool = False):
        self.path = db_path
        self.stream = stream
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA query_only = 1")
        self._cache: dict[str, np.ndarray] = {}
        self.bytes_read = 0
        self.reads = 0
        self._shapes = {
            name: (shape, dtype)
            for name, shape, dtype in self.conn.execute(
                "SELECT name, shape, dtype FROM tensors"
            )
        }

    def has(self, name: str) -> bool:
        return name in self._shapes

    def get(self, name: str) -> np.ndarray:
        """A tensor as float32, in numpy's row-major orientation.

        reminis stores `shape` reversed relative to the data layout, which is
        GGUF's convention, so reversing it back is what turns the blob into
        the matrix the maths expects: a linear layer's weight comes out
        (out_features, in_features).
        """
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        row = self.conn.execute(
            "SELECT shape, dtype, data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise UnsupportedModel(f"This model has no tensor named '{name}'")
        shape, dtype, blob = row
        if not is_float_dtype(dtype):
            raise UnsupportedModel(
                f"'{name}' is stored as {dtype}, a quantized type. Running a "
                f"model needs float weights -- convert an F16/F32 GGUF or a "
                f"safetensors checkpoint instead."
            )

        arr = to_float32(blob, dtype).reshape(tuple(ast.literal_eval(shape))[::-1])
        self.bytes_read += len(blob)
        self.reads += 1
        if not self.stream:
            self._cache[name] = arr
        return arr

    def close(self):
        self.conn.close()


class Config:
    """The hyperparameters, read out of `model_meta`."""

    def __init__(self, meta: dict, store: WeightStore):
        self.arch = meta.get("general.architecture", "")
        if self.arch not in SUPPORTED_ARCHS:
            raise UnsupportedModel(
                f"reminis run implements {', '.join(sorted(SUPPORTED_ARCHS))}; "
                f"this model's architecture is '{self.arch or 'unknown'}'.\n"
                f"Inspect it instead:  reminis view <db>"
            )
        self.rope_style = SUPPORTED_ARCHS[self.arch]
        a = self.arch

        def num(key, default=None):
            value = meta.get(f"{a}.{key}")
            if value is None:
                if default is None:
                    raise UnsupportedModel(f"Model metadata is missing {a}.{key}")
                return default
            return float(value) if "." in str(value) or "e" in str(value).lower() else int(value)

        self.n_layers = int(num("block_count"))
        self.d_model = int(num("embedding_length"))
        self.n_heads = int(num("attention.head_count"))
        self.n_kv_heads = int(num("attention.head_count_kv", self.n_heads))
        self.head_dim = int(num("attention.key_length", self.d_model // self.n_heads))
        self.rope_base = float(num("rope.freq_base", 10000.0))
        self.rms_eps = float(num("attention.layer_norm_rms_epsilon", 1e-5))
        self.context_length = int(num("context_length", 2048))
        self.rope_dim = int(num("rope.dimension_count", self.head_dim))

        if self.n_heads % self.n_kv_heads != 0:
            raise UnsupportedModel(
                f"{self.n_heads} attention heads do not divide evenly into "
                f"{self.n_kv_heads} key/value heads"
            )

        # Llama 3 stores per-dimension rotary scaling as a tensor rather than
        # as metadata. Ignoring it silently would break long-context models
        # in a way that only shows up as slightly-wrong text.
        self.rope_factors = (
            store.get("rope_freqs.weight").ravel()
            if store.has("rope_freqs.weight") else None
        )

        # Tied embeddings: many small models have no separate output matrix.
        self.tied_output = not store.has("output.weight")


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
_PRETOKENIZERS["qwen2"] = _PRETOKENIZERS["llama-bpe"]
_PRETOKENIZERS["smollm"] = _PRETOKENIZERS["default"]
_PRETOKENIZERS["gpt-2"] = _PRETOKENIZERS["default"]


class Tokenizer:
    """Byte-level BPE, rebuilt from the vocabulary and merges in the database."""

    def __init__(self, meta: dict):
        import re

        model = meta.get("tokenizer.ggml.model")
        if model != "gpt2":
            raise UnsupportedModel(
                f"This model's tokenizer is '{model or 'missing'}'. reminis run "
                f"implements byte-level BPE (the 'gpt2' tokenizer), which is "
                f"what llama, qwen2 and smollm use."
            )

        self.tokens = _parse_array(meta, "tokenizer.ggml.tokens")
        merges = _parse_array(meta, "tokenizer.ggml.merges")
        if not self.tokens or not merges:
            raise UnsupportedModel(
                "The database has no tokenizer vocabulary in it, so there is "
                "nothing to encode with."
            )

        self.ids = {t: i for i, t in enumerate(self.tokens)}
        self.ranks = {tuple(m.split(" ")): i for i, m in enumerate(merges)}

        # GGUF token type 3 is CONTROL: <|im_start|> and friends, which must
        # be matched whole rather than split into bytes. Numeric GGUF arrays
        # come back as one-element lists per entry, since that is how the
        # reader hands them over.
        types = [
            t[0] if isinstance(t, (list, tuple)) else t
            for t in _parse_array(meta, "tokenizer.ggml.token_type")
        ]
        self.specials = sorted(
            (self.tokens[i] for i, t in enumerate(types) if int(t) == 3),
            key=len, reverse=True,
        )

        pre = meta.get("tokenizer.ggml.pre", "default")
        self.pattern = re.compile(_PRETOKENIZERS.get(pre, _PRETOKENIZERS["default"]))
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(s) for s in self.specials) + ")")
            if self.specials else None
        )
        self._re = re

        self.bos_id = _int_or_none(meta.get("tokenizer.ggml.bos_token_id"))
        self.eos_id = _int_or_none(meta.get("tokenizer.ggml.eos_token_id"))
        self.add_bos = str(meta.get("tokenizer.ggml.add_bos_token", "False")).lower() == "true"
        self.chat_template = meta.get("tokenizer.chat_template", "")

        self._byte_to_char, self._char_to_byte = _byte_unicode_maps()
        self._bpe_cache: dict[str, list[str]] = {}

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


def _parse_array(meta: dict, key: str) -> list:
    """Read one of the list-shaped metadata values back into a Python list.

    The converter stores GGUF arrays as their Python repr, so this is the
    inverse. `literal_eval` rather than `eval`: these strings come out of a
    file someone downloaded.
    """
    raw = meta.get(key)
    if not raw:
        return []
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise UnsupportedModel(f"Could not read {key} from this database: {exc}")


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- the model


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    var = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True)
    return (x * np.reciprocal(np.sqrt(var + eps))) * weight


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    np.exp(x, out=x)
    return x / np.sum(x, axis=axis, keepdims=True)


class Model:
    """A llama-family transformer whose weights come from SQLite."""

    def __init__(self, db_path: str, stream: bool = False):
        self.store = WeightStore(db_path, stream=stream)
        meta = dict(self.store.conn.execute("SELECT key, value FROM model_meta"))
        self.meta = meta
        self.cfg = Config(meta, self.store)
        self.tokenizer = Tokenizer(meta)
        self._rope_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def close(self):
        self.store.close()

    # -- rotary embeddings ------------------------------------------------

    def _rope_tables(self, n_positions: int, offset: int):
        """cos/sin for positions [offset, offset + n_positions)."""
        cfg = self.cfg
        half = cfg.rope_dim // 2
        inv_freq = 1.0 / (
            cfg.rope_base ** (np.arange(half, dtype=np.float32) * 2.0 / cfg.rope_dim)
        )
        if cfg.rope_factors is not None:
            inv_freq = inv_freq / cfg.rope_factors[:half]
        pos = np.arange(offset, offset + n_positions, dtype=np.float32)
        angles = pos[:, None] * inv_freq[None, :]
        return np.cos(angles), np.sin(angles)

    def _apply_rope(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """Rotate the first `rope_dim` channels of each head.

        Two layouts exist and they are not interchangeable. llama.cpp's
        "norm" rotates adjacent channels as pairs; "neox" rotates channel i
        against channel i + rope_dim/2. Which one a model wants depends on
        how its weights were laid out at conversion time, so it comes from
        the architecture rather than from a guess.
        """
        d = self.cfg.rope_dim
        rot, rest = x[..., :d], x[..., d:]
        cos = cos[:, None, :]
        sin = sin[:, None, :]

        if self.cfg.rope_style == "norm":
            even, odd = rot[..., 0::2], rot[..., 1::2]
            out = np.empty_like(rot)
            out[..., 0::2] = even * cos - odd * sin
            out[..., 1::2] = even * sin + odd * cos
        else:
            half = d // 2
            first, second = rot[..., :half], rot[..., half:]
            out = np.concatenate(
                [first * cos - second * sin, first * sin + second * cos], axis=-1
            )

        return out if rest.size == 0 else np.concatenate([out, rest], axis=-1)

    # -- one layer --------------------------------------------------------

    def _linear(self, x: np.ndarray, name: str, bias: str | None = None) -> np.ndarray:
        y = x @ self.store.get(name).T
        if bias and self.store.has(bias):
            y = y + self.store.get(bias).ravel()
        return y

    def _block(self, x: np.ndarray, layer: int, cache: "KVCache", offset: int):
        cfg = self.cfg
        p = f"blk.{layer}."
        n_tokens = x.shape[0]

        h = _rms_norm(x, self.store.get(p + "attn_norm.weight").ravel(), cfg.rms_eps)

        q = self._linear(h, p + "attn_q.weight", p + "attn_q.bias")
        k = self._linear(h, p + "attn_k.weight", p + "attn_k.bias")
        v = self._linear(h, p + "attn_v.weight", p + "attn_v.bias")

        q = q.reshape(n_tokens, cfg.n_heads, cfg.head_dim)
        k = k.reshape(n_tokens, cfg.n_kv_heads, cfg.head_dim)
        v = v.reshape(n_tokens, cfg.n_kv_heads, cfg.head_dim)

        cos, sin = self._rope_tables(n_tokens, offset)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        k_all, v_all = cache.append(layer, k, v)
        total = k_all.shape[0]

        # Grouped-query attention: each key/value head serves several query
        # heads, so the cache is repeated rather than stored per query head.
        repeat = cfg.n_heads // cfg.n_kv_heads
        if repeat > 1:
            k_all = np.repeat(k_all, repeat, axis=1)
            v_all = np.repeat(v_all, repeat, axis=1)

        # (heads, tokens, dim) so the per-head matmuls batch.
        qh = q.transpose(1, 0, 2)
        kh = k_all.transpose(1, 2, 0)
        vh = v_all.transpose(1, 0, 2)

        scores = (qh @ kh) * (1.0 / math.sqrt(cfg.head_dim))
        # Causal mask, written against absolute positions so it is correct
        # both for a batched prompt and for one token against a long cache.
        rows = np.arange(offset, offset + n_tokens)[:, None]
        cols = np.arange(total)[None, :]
        scores = np.where(cols <= rows, scores, np.float32(-np.inf))

        attn = _softmax(scores.astype(np.float32))
        out = (attn @ vh).transpose(1, 0, 2).reshape(n_tokens, cfg.n_heads * cfg.head_dim)
        x = x + self._linear(out, p + "attn_output.weight", p + "attn_output.bias")

        h = _rms_norm(x, self.store.get(p + "ffn_norm.weight").ravel(), cfg.rms_eps)
        gate = _silu(self._linear(h, p + "ffn_gate.weight"))
        up = self._linear(h, p + "ffn_up.weight")
        return x + self._linear(gate * up, p + "ffn_down.weight")

    def forward(self, tokens: list[int], cache: "KVCache", offset: int) -> np.ndarray:
        """Logits for the last token of `tokens`.

        Floating-point warnings are silenced for the duration. Two of them
        would otherwise fire on every token for reasons that are not bugs:
        the causal mask is built out of -inf on purpose, and Apple's
        Accelerate BLAS raises the divide-by-zero and overflow flags during
        perfectly ordinary float32 matmuls whose inputs and outputs are all
        finite. The logits are checked explicitly below instead, which
        catches a genuinely broken forward pass without the noise.
        """
        with np.errstate(all="ignore"):
            embed = self.store.get("token_embd.weight")
            x = embed[np.asarray(tokens, dtype=np.int64)].astype(np.float32)

            for layer in range(self.cfg.n_layers):
                x = self._block(x, layer, cache, offset)

            x = _rms_norm(x, self.store.get("output_norm.weight").ravel(), self.cfg.rms_eps)
            last = x[-1:]
            out_name = "token_embd.weight" if self.cfg.tied_output else "output.weight"
            logits = (last @ self.store.get(out_name).T).ravel()

        if not np.isfinite(logits).all():
            raise ValueError(
                "The forward pass produced non-finite logits. The weights in "
                "this database do not form a working model."
            )
        return logits


class KVCache:
    """Keys and values for every layer, grown a token at a time."""

    def __init__(self, n_layers: int):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def append(self, layer: int, k: np.ndarray, v: np.ndarray):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = np.concatenate([self.k[layer], k], axis=0)
            self.v[layer] = np.concatenate([self.v[layer], v], axis=0)
        return self.k[layer], self.v[layer]

    @property
    def length(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[0]


# ---------------------------------------------------------------- sampling


def _sample(logits: np.ndarray, temperature: float, top_p: float, rng) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))

    probs = _softmax(logits.astype(np.float32) / np.float32(temperature))

    if 0 < top_p < 1:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        # Keep the smallest set of tokens whose mass reaches top_p, always
        # including the first one so the set is never empty.
        keep = int(np.searchsorted(cumulative, top_p) + 1)
        chosen = order[:keep]
        renormalised = probs[chosen] / probs[chosen].sum()
        return int(rng.choice(chosen, p=renormalised))

    return int(rng.choice(len(probs), p=probs))


def generate(
    db_path: str,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int | None = None,
    stream: bool = False,
    chat: bool = False,
    stop_at_eos: bool = True,
    verbose: bool = True,
    on_token=None,
) -> dict:
    """Generate text from a model stored in a reminis database.

    Args:
        db_path: The model database.
        prompt: The prompt text.
        max_tokens: How many tokens to generate at most.
        temperature: 0 is greedy; higher is more random.
        top_p: Nucleus sampling cutoff. 1 disables it.
        seed: Seed for sampling, so a run can be repeated exactly.
        stream: Re-read every weight from SQLite instead of caching it, so
            peak memory is one layer rather than the whole model.
        chat: Wrap the prompt in the model's chat template, when it has a
            ChatML one.
        stop_at_eos: Stop when the model emits its end-of-text token.
        verbose: Print the header, the prompt, and the closing timings. The
            generated text itself is printed either way, so that piping the
            output somewhere gives the completion and nothing else.
        on_token: Optional callback, called with each decoded token string.
            Supplying one turns off printing, since the caller is handling it.

    Returns:
        A dict with the prompt, the completion, and timing.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    model = Model(db_path, stream=stream)
    tok = model.tokenizer
    rng = np.random.default_rng(seed)

    try:
        text = _apply_chat_template(prompt, tok) if chat else prompt
        tokens = tok.encode(text)
        if not tokens:
            raise ValueError("The prompt encoded to zero tokens")
        if len(tokens) >= model.cfg.context_length:
            raise ValueError(
                f"The prompt is {len(tokens)} tokens and this model's context "
                f"is {model.cfg.context_length}"
            )

        if verbose:
            mode = "streaming from SQLite" if stream else "weights cached in RAM"
            print(f"{model.meta.get('general.name', Path(db_path).name)} "
                  f"| {model.cfg.arch} | {model.cfg.n_layers} layers | {mode}")
            print(f"{len(tokens)} prompt tokens\n")
            print(text, end="", flush=True)

        cache = KVCache(model.cfg.n_layers)

        t0 = time.time()
        logits = model.forward(tokens, cache, offset=0)
        prefill_seconds = time.time() - t0

        produced = []
        t1 = time.time()
        for _ in range(max_tokens):
            token_id = _sample(logits, temperature, top_p, rng)
            if stop_at_eos and tok.eos_id is not None and token_id == tok.eos_id:
                break
            produced.append(token_id)
            piece = tok.decode_one(token_id)
            if on_token:
                on_token(piece)
            else:
                print(piece, end="", flush=True)
            logits = model.forward([token_id], cache, offset=cache.length)

        decode_seconds = time.time() - t1
        completion = tok.decode(produced)

        if not on_token:
            print()
        if verbose:
            rate = len(produced) / decode_seconds if decode_seconds else 0
            print(f"\n{len(tokens)} prompt tokens in {prefill_seconds:.2f}s, "
                  f"{len(produced)} generated in {decode_seconds:.2f}s "
                  f"({rate:.1f} tok/s)")
            read = model.store.bytes_read / (1024 ** 2)
            if stream:
                print(f"Read {read:,.0f} MB from SQLite across "
                      f"{model.store.reads:,} queries, cached nothing")
            else:
                print(f"Read {read:,.0f} MB from SQLite once, then reused it")

        return {
            "prompt": text,
            "completion": completion,
            "prompt_tokens": len(tokens),
            "generated_tokens": len(produced),
            "token_ids": produced,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "bytes_read": model.store.bytes_read,
            "queries": model.store.reads,
        }
    finally:
        model.close()


def _apply_chat_template(prompt: str, tok: Tokenizer) -> str:
    """Wrap a prompt as a chat turn, for models that use ChatML.

    The stored template is Jinja, and rendering Jinja to run one prompt is
    more machinery than it is worth. ChatML is recognisable and it is what
    the small instruct models here use; anything else is left alone rather
    than mangled into a format the model was not trained on.
    """
    if "<|im_start|>" not in (tok.chat_template or ""):
        return prompt
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )


def run_cli(args, on_error=None):
    """Entry point for `reminis run`, kept here so the CLI stays thin."""
    try:
        generate(
            args.input, args.prompt,
            max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p,
            seed=args.seed, stream=args.stream, chat=args.chat,
            verbose=not args.quiet,
        )
    except (UnsupportedModel, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
