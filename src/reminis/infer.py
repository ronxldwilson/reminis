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
expected -- 0.86-0.89x llama.cpp's CPU token generation across three model
sizes, and faster than it at prompt processing, since both end up in the
platform BLAS for the large matrix multiplies. Against llama.cpp on the GPU
it is 2.6-2.8x slower.

``--stream`` takes that further. In streaming mode no weight is ever cached:
every matrix multiplication in every layer re-reads its operand from SQLite
and throws it away, so peak memory is one layer rather than one model. It is
slow, and it demonstrates the thing the whole project is about -- that the
model is data in a database, paged in on demand, not a file that must fit in
RAM.

Scope, deliberately narrow and loudly enforced:

  * llama-family and qwen2 architectures from GGUF (rotary, RMSNorm, SwiGLU,
    grouped-query attention), which covers llama, qwen2, smollm, mistral
  * float weights -- F32, F16, BF16
  * quantized weights, unpacked at load through the ``gguf`` package: every
    K-quant and i-quant llama.cpp writes. Note what this is not -- the blocks
    become float16 in memory, so a quantized model becomes *runnable*, not
    *small*. Quantization saves the file and the download here, not the RAM.
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

from reminis.backend import select as select_backend
from reminis.dtypes import (
    dequantize_to_float32,
    is_float_dtype,
    is_quantized_dtype,
)

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

    # Weights worth keeping packed: the per-layer matrices, which are almost
    # all of a model's bytes and are only ever used as the right-hand side of
    # a matrix multiply. Embeddings are indexed row-wise rather than
    # multiplied, and norms are tiny, so neither is packed.
    _PACKABLE = (
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    )

    def __init__(self, db_path: str, stream: bool = False, backend=None,
                 pack_bits: int | None = None, pack_group: int = 32):
        self.path = db_path
        self.stream = stream
        self.backend = backend or select_backend("inference")
        self.pack_bits = pack_bits
        self.pack_group = pack_group
        self.packed = 0
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA query_only = 1")
        # Memory-map the file rather than copying each blob through SQLite's
        # own buffer. Measured on a 258 MB model, reading every weight goes
        # from 4.1 GB/s to 6.7 GB/s, which shows up in load time and in every
        # single read that streaming mode does. The size is a ceiling, not an
        # allocation: SQLite maps what the file actually needs.
        self.conn.execute("PRAGMA mmap_size = 34359738368")
        self._cache: dict[str, np.ndarray] = {}
        self.bytes_read = 0
        self.reads = 0
        self.dequantized = 0
        self._shapes = {
            name: (shape, dtype)
            for name, shape, dtype in self.conn.execute(
                "SELECT name, shape, dtype FROM tensors"
            )
        }

    def has(self, name: str) -> bool:
        return name in self._shapes

    def get(self, name: str):
        """A tensor in the backend's compute dtype, row-major.

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
        dims = tuple(ast.literal_eval(shape))[::-1]

        if is_float_dtype(dtype):
            arr = self.backend.from_bytes(blob, dtype, dims)
        elif is_quantized_dtype(dtype):
            # Quantized blocks are unpacked to float32 and then handed to the
            # backend, which narrows them to its compute dtype. This costs
            # more memory than the file did -- the whole point of quantizing
            # is undone -- and it is what makes a quantized model runnable at
            # all rather than a hard refusal.
            self.dequantized += 1
            arr = self.backend.from_numpy(
                dequantize_to_float32(blob, dtype).reshape(dims)
            )
            # Unpacking produces float32, and the backend narrows it to its
            # own compute dtype. On a lazy backend both would stay alive
            # until something forced the issue, so the float32 copy of every
            # weight in the model would accumulate. Force it here.
            self.backend.eval(arr)
        else:
            raise UnsupportedModel(
                f"'{name}' is stored as {dtype}, which is neither a float type "
                f"nor a quantization reminis can unpack."
            )
        if self._should_pack(name):
            arr = self.backend.pack(arr, self.pack_bits, self.pack_group)
            self.packed += 1

        self.bytes_read += len(blob)
        self.reads += 1
        if not self.stream:
            self._cache[name] = arr
        return arr

    def _should_pack(self, name: str) -> bool:
        return (
            self.pack_bits is not None
            and self.backend.can_pack()
            and name.startswith("blk.")
            and name.endswith(self._PACKABLE)
        )

    def get_numpy(self, name: str) -> np.ndarray:
        """A tensor as plain float32 numpy, whatever the backend is.

        For the handful of values that are not part of the arithmetic on the
        hot path -- rotary scaling factors, say -- and are used to build
        tables in full precision.
        """
        row = self.conn.execute(
            "SELECT dtype, data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise UnsupportedModel(f"This model has no tensor named '{name}'")
        from reminis.dtypes import to_float32_any

        return to_float32_any(row[1], row[0])

    def drop(self, name: str):
        """Forget a cached tensor, once something else stands in for it."""
        self._cache.pop(name, None)

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
            store.get_numpy("rope_freqs.weight").ravel()
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


class Model:
    """A llama-family transformer whose weights come from SQLite.

    The arithmetic is written against a backend rather than against numpy
    directly, so the same forward pass runs on the CPU or on a GPU depending
    on what the machine has. numpy stays the reference: it is the one whose
    logits are checked against transformers to five decimal places, and the
    others are checked against it.
    """

    def __init__(self, db_path: str, stream: bool = False, backend=None,
                 pack_bits: int | None = None):
        self.backend = backend or select_backend("inference")
        self.store = WeightStore(db_path, stream=stream, backend=self.backend,
                                 pack_bits=pack_bits)
        meta = dict(self.store.conn.execute("SELECT key, value FROM model_meta"))
        self.meta = meta
        self.cfg = Config(meta, self.store)
        self.tokenizer = Tokenizer(meta)
        self._rope_cache: tuple | None = None
        self._mask_cache: tuple | None = None
        self._fused_cache: dict[tuple[int, str], tuple] = {}

    def close(self):
        self.store.close()

    # -- rotary embeddings ------------------------------------------------

    def _rope_tables(self, n_positions: int, offset: int):
        """cos/sin for positions [offset, offset + n_positions).

        Every layer in a forward pass asks for the same positions, so the
        trigonometry was being redone once per layer -- thirty times per
        token on a 135M model. One entry is enough to hold it.
        """
        if self._rope_cache is not None:
            key_n, key_offset, cos, sin = self._rope_cache
            if key_n == n_positions and key_offset == offset:
                return cos, sin

        cfg = self.cfg
        half = cfg.rope_dim // 2
        # The tables are built in numpy at full precision regardless of the
        # backend. They are tiny, they are cached, and the angles are the one
        # place where half precision would visibly cost accuracy: an error in
        # a position's angle shifts every token that attends to it.
        inv_freq = 1.0 / (
            cfg.rope_base ** (np.arange(half, dtype=np.float32) * 2.0 / cfg.rope_dim)
        )
        if cfg.rope_factors is not None:
            inv_freq = inv_freq / cfg.rope_factors[:half]
        pos = np.arange(offset, offset + n_positions, dtype=np.float32)
        angles = pos[:, None] * inv_freq[None, :]
        cos = self.backend.from_numpy(np.cos(angles))
        sin = self.backend.from_numpy(np.sin(angles))
        self._rope_cache = (n_positions, offset, cos, sin)
        return cos, sin

    def _apply_rope(self, x, cos, sin):
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

        xp = self.backend.xp
        if self.cfg.rope_style == "norm":
            even, odd = rot[..., 0::2], rot[..., 1::2]
            # Interleaving the two halves back together by stacking and
            # reshaping rather than by assigning into a buffer, since not
            # every backend's arrays can be written into piecewise.
            rotated = xp.stack([even * cos - odd * sin, even * sin + odd * cos],
                               axis=-1)
            out = rotated.reshape(rot.shape)
        else:
            half = d // 2
            first, second = rot[..., :half], rot[..., half:]
            out = xp.concatenate(
                [first * cos - second * sin, first * sin + second * cos], axis=-1
            )

        return out if rest.shape[-1] == 0 else xp.concatenate([out, rest], axis=-1)

    # -- one layer --------------------------------------------------------

    def _linear(self, x, name: str, bias: str | None = None):
        y = self.backend.matmul_weight(x, self.store.get(name))
        if bias and self.store.has(bias):
            y = y + self.store.get(bias).reshape(-1)
        return y

    def _fused(self, layer: int, key: str, names: list[str]):
        """One weight matrix standing in for several stacked ones.

        Q, K and V are three separate matrix-vector products reading three
        separate regions of memory, and so are the FFN's gate and up. Stacking
        each group into a single matrix turns them into one call over one
        contiguous region. Decoding is memory-bound, so what this buys is not
        arithmetic but streaming: measured on the real shapes, the fused QKV
        is about a third faster than the three separate ones.

        The originals are dropped from the cache as the fused copy is built,
        so this costs no extra memory. It is skipped entirely in streaming
        mode, where caching anything would defeat the point.
        """
        cached = self._fused_cache.get((layer, key))
        if cached is not None:
            return cached

        xp = self.backend.xp
        prefix = f"blk.{layer}."
        weight = self.backend.contiguous(
            xp.concatenate([self.store.get(prefix + n) for n in names], axis=0)
        )
        biases = [prefix + n.replace(".weight", ".bias") for n in names]
        bias = None
        if all(self.store.has(b) for b in biases):
            bias = xp.concatenate([self.store.get(b).reshape(-1) for b in biases])

        for n in names:
            self.store.drop(prefix + n)
        entry = (weight, bias)
        self._fused_cache[(layer, key)] = entry
        return entry

    def _qkv(self, h: np.ndarray, layer: int, prefix: str):
        # Packed weights cannot be stacked into one matrix -- concatenating
        # them would mean unpacking, which is the thing being avoided -- so
        # fusion and packing are alternatives, not companions.
        if self.store.stream or self.store.pack_bits is not None:
            return (self._linear(h, prefix + "attn_q.weight", prefix + "attn_q.bias"),
                    self._linear(h, prefix + "attn_k.weight", prefix + "attn_k.bias"),
                    self._linear(h, prefix + "attn_v.weight", prefix + "attn_v.bias"))

        weight, bias = self._fused(
            layer, "qkv",
            ["attn_q.weight", "attn_k.weight", "attn_v.weight"],
        )
        out = h @ weight.T
        if bias is not None:
            out = out + bias
        cut_q = self.cfg.n_heads * self.cfg.head_dim
        cut_k = cut_q + self.cfg.n_kv_heads * self.cfg.head_dim
        return out[:, :cut_q], out[:, cut_q:cut_k], out[:, cut_k:]

    def _gate_up(self, h: np.ndarray, layer: int, prefix: str):
        if self.store.stream or self.store.pack_bits is not None:
            return (self._linear(h, prefix + "ffn_gate.weight"),
                    self._linear(h, prefix + "ffn_up.weight"))

        weight, _ = self._fused(layer, "gate_up", ["ffn_gate.weight", "ffn_up.weight"])
        out = h @ weight.T
        half = out.shape[1] // 2
        return out[:, :half], out[:, half:]

    def _block(self, x, layer: int, cache: "KVCache", offset: int):
        cfg = self.cfg
        b = self.backend
        xp = b.xp
        p = f"blk.{layer}."
        n_tokens = x.shape[0]

        h = b.rms_norm(x, self.store.get(p + "attn_norm.weight").reshape(-1), cfg.rms_eps)

        q, k, v = self._qkv(h, layer, p)
        q = q.reshape(n_tokens, cfg.n_heads, cfg.head_dim)
        k = k.reshape(n_tokens, cfg.n_kv_heads, cfg.head_dim)
        v = v.reshape(n_tokens, cfg.n_kv_heads, cfg.head_dim)

        cos, sin = self._rope_tables(n_tokens, offset)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        k_all, v_all = cache.append(layer, k, v)
        total = k_all.shape[0]

        # Grouped-query attention. Each key/value head serves `repeat` query
        # heads, and the obvious way to write that is np.repeat on the cache
        # -- which copies the whole cache, every layer, every token. Adding a
        # length-1 axis instead lets broadcasting do it for free: queries are
        # grouped (n_kv, repeat, ...) and the keys and values broadcast across
        # the group they belong to.
        repeat = cfg.n_heads // cfg.n_kv_heads
        qh = q.transpose(1, 0, 2).reshape(cfg.n_kv_heads, repeat, n_tokens, cfg.head_dim)
        kh = k_all.transpose(1, 2, 0)[:, None]
        vh = v_all.transpose(1, 0, 2)[:, None]

        scores = (qh @ kh) * (1.0 / math.sqrt(cfg.head_dim))
        # A single token attends to the whole cache, so there is nothing for
        # a causal mask to hide and building one is pure waste. It is only
        # needed when several tokens are processed at once.
        if n_tokens > 1:
            mask = self._causal_mask(n_tokens, offset, total)
            scores = xp.where(mask, scores, float("-inf"))

        attn = b.softmax(scores)
        out = (attn @ vh).reshape(cfg.n_heads, n_tokens, cfg.head_dim)
        out = out.transpose(1, 0, 2).reshape(n_tokens, cfg.n_heads * cfg.head_dim)
        x = x + self._linear(out, p + "attn_output.weight", p + "attn_output.bias")

        h = b.rms_norm(x, self.store.get(p + "ffn_norm.weight").reshape(-1), cfg.rms_eps)
        gate, up = self._gate_up(h, layer, p)
        return x + self._linear(b.silu(gate) * up, p + "ffn_down.weight")

    def _causal_mask(self, n_tokens: int, offset: int, total: int):
        """True where a query at an absolute position may see a key.

        Built in numpy and handed to the backend, because it depends only on
        positions and is reused unchanged by every layer of the pass.
        """
        key = (n_tokens, offset, total)
        if self._mask_cache is not None and self._mask_cache[0] == key:
            return self._mask_cache[1]
        rows = np.arange(offset, offset + n_tokens)[:, None]
        cols = np.arange(total)[None, :]
        mask = self.backend.xp.array(cols <= rows)
        self._mask_cache = (key, mask)
        return mask

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
        b = self.backend
        with b.errstate():
            embed = self.store.get("token_embd.weight")
            x = embed[b.xp.array(np.asarray(tokens, dtype=np.int32))]

            for layer in range(self.cfg.n_layers):
                x = self._block(x, layer, cache, offset)

            x = b.rms_norm(x, self.store.get("output_norm.weight").reshape(-1),
                           self.cfg.rms_eps)
            last = x[-1:]
            out_name = "token_embd.weight" if self.cfg.tied_output else "output.weight"
            logits = self.backend.matmul_weight(
                last, self.store.get(out_name)
            ).reshape(-1)
            b.eval(logits)

        # Sampling happens in numpy whatever the backend: the vocabulary-sized
        # vector is small, the operations on it are sequential, and keeping
        # one code path means a seed reproduces the same text everywhere.
        logits = b.to_numpy(logits)

        if not np.isfinite(logits).all():
            raise ValueError(
                "The forward pass produced non-finite logits. The weights in "
                "this database do not form a working model."
            )
        return logits


class KVCache:
    """Keys and values for every layer.

    Growing this with `np.concatenate` reallocates and recopies the entire
    cache on every token, which turns a linear cost into a quadratic one and
    is invisible until the context is long. Given a capacity up front it
    allocates once and writes each token into place, returning a view.
    Without one it still grows, in doubling steps rather than by one.
    """

    def __init__(self, n_layers: int, capacity: int | None = None, backend=None):
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.capacity = capacity
        self.backend = backend or select_backend("inference")
        self._used = 0

    def _empty(self, size, tail, like):
        xp = self.backend.xp
        return xp.zeros((size,) + tuple(tail), dtype=like.dtype)

    def append(self, layer: int, k, v):
        n = k.shape[0]
        buf_k, buf_v = self.k[layer], self.v[layer]

        if buf_k is None:
            size = max(self.capacity or 0, n)
            buf_k = self._empty(size, k.shape[1:], k)
            buf_v = self._empty(size, v.shape[1:], v)
            self.k[layer], self.v[layer] = buf_k, buf_v
            used = 0
        else:
            used = self._used
            if used + n > buf_k.shape[0]:
                grown = max(buf_k.shape[0] * 2, used + n)
                bigger_k = self._empty(grown, k.shape[1:], k)
                bigger_v = self._empty(grown, v.shape[1:], v)
                bigger_k[:used] = buf_k[:used]
                bigger_v[:used] = buf_v[:used]
                buf_k, buf_v = bigger_k, bigger_v
                self.k[layer], self.v[layer] = buf_k, buf_v

        buf_k[used:used + n] = k
        buf_v[used:used + n] = v

        # The counter advances once per token, not once per layer, so it is
        # updated on the last layer only -- every layer sees the same span.
        if layer == len(self.k) - 1:
            self._used = used + n
        return buf_k[:used + n], buf_v[:used + n]

    @property
    def length(self) -> int:
        return self._used


# ---------------------------------------------------------------- sampling


def _sample(logits: np.ndarray, temperature: float, top_p: float, rng) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))

    scaled = logits.astype(np.float32) / np.float32(temperature)
    scaled = scaled - np.max(scaled)
    np.exp(scaled, out=scaled)
    probs = scaled / np.sum(scaled)

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
    backend: str | None = None,
    pack_bits: int | None = None,
) -> dict:
    """Generate text from a model stored in a reminis database.

    Args:
        db_path: The model database.
        prompt: The prompt text.
        max_tokens: How many tokens to generate at most.
        temperature: 0 is greedy; higher is more random.
        top_p: Nucleus sampling cutoff. 1 disables it.
        seed: Seed for sampling, so a run can be repeated exactly.
        pack_bits: Keep the big per-layer matrices packed at this many bits
            instead of unpacking them to float16, on a backend that can
            multiply them packed. Trades accuracy for memory: 6 keeps the
            top-5 ranking intact for 1.7x less, 4 goes to 2.1x less and
            visibly reorders it.
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

    chosen = select_backend("inference", backend)
    model = Model(db_path, stream=stream, backend=chosen, pack_bits=pack_bits)
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
                  f"| {model.cfg.arch} | {model.cfg.n_layers} layers | "
                  f"{chosen.describe()} | {mode}")
            print(f"{len(tokens)} prompt tokens\n")
            print(text, end="", flush=True)

        cache = KVCache(model.cfg.n_layers, capacity=len(tokens) + max_tokens,
                        backend=chosen)

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
            "backend": chosen.name,
            "packed_tensors": model.store.packed,
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
            verbose=not args.quiet, backend=args.backend,
            pack_bits=args.pack,
        )
    except (UnsupportedModel, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
