"""Repack GGML quantization blocks into a backend's affine format, losslessly.

A quantized model normally has to be unpacked to float16 before anything
here can multiply by it, which throws away the memory saving that was the
whole reason to quantize. Re-quantizing into the backend's own format keeps
the saving but rounds every weight a second time -- measured at 7.8% extra
relative error for 4-bit, enough to reorder a model's top-5 tokens.

There is a third option for some types, and it costs nothing. Several GGML
quantizations are *already* affine within each group of 32 weights:

    Q4_0   w = d * (q - 8)                   one group of 32 per block
    Q4_1   w = d * q + m                     one group of 32 per block
    Q5_0   w = d * (q - 16)                  one group of 32 per block
    Q5_1   w = d * q + m                     one group of 32 per block
    Q8_0   w = d * q                         one group of 32 per block
    Q4_K   w = (d * sc) * q - (dmin * m)     8 groups of 32 per superblock
    Q5_K   w = (d * sc) * q - (dmin * m)     8 groups of 32 per superblock

which is exactly the form ``w = scale * q + bias`` that mlx's quantized
matmul expects. So they can be rewritten into its layout by moving bits
around -- no arithmetic on the weights at all, and no second rounding. The
scales are held in float32 on that side, so even the products survive
exactly.

What does not map:

    Q6_K, Q2_K, Q3_K   sub-blocks of 16 weights, and mlx supports groups of
                       32, 64 or 128 only
    IQ2..IQ4           codebook types; the values are table lookups rather
                       than a scale times an integer, so no affine form exists

Those fall back to being unpacked, which is what happened to everything
before this existed.
"""

import numpy as np

# Block size in bytes, weights per block, and bits per weight, for the
# quantizations that have an exact affine form at group size 32.
_LAYOUT = {
    "Q4_0": (18, 32, 4),
    "Q4_1": (20, 32, 4),
    "Q5_0": (22, 32, 5),
    "Q5_1": (24, 32, 5),
    "Q8_0": (34, 32, 8),
    "Q4_K": (144, 256, 4),
    "Q5_K": (176, 256, 5),
}

AFFINE_GROUP = 32

# For quantizations with no exact affine form, the width to re-quantize to.
# Chosen at or just above the source's own precision, so the second rounding
# adds as little as it can while still being far smaller than float16. This
# path is lossy, unlike `repack`, and the two are counted separately so a run
# can say how much of it was exact.
NEAREST_BITS = {
    "Q2_K": 3, "Q3_K": 4, "Q3_K_S": 4, "Q3_K_M": 4, "Q3_K_L": 4,
    "Q6_K": 6,
    "IQ1_S": 2, "IQ1_M": 2, "IQ2_XXS": 2, "IQ2_XS": 3, "IQ2_S": 3,
    "IQ3_XXS": 4, "IQ3_S": 4, "IQ3_M": 4, "IQ4_NL": 4, "IQ4_XS": 4,
    # MXFP4 shares an exponent across 32 values and stores each as a tiny
    # float, so its levels are not evenly spaced and no affine form exists.
    "MXFP4": 4, "NVFP4": 4,
}


def nearest_bits(dtype: str):
    """The width to re-quantize a non-affine type to, or None to leave it."""
    return NEAREST_BITS.get(dtype)


def can_repack(dtype: str) -> bool:
    """Whether this quantization has an exact affine form at group size 32."""
    return dtype in _LAYOUT


def _q4_k_scale_min(scales: np.ndarray):
    """The 8 sub-block scales and mins packed into Q4_K's 12 bytes.

    Six bits each, spread across the bytes in a pattern that is easier to
    read as a diagram than as prose -- see the comment in ``gguf.quants``.
    This mirrors that unpacking exactly, because being off by one bit here
    would produce a model that still speaks and is subtly wrong.
    """
    scales = scales.reshape((-1, 3, 4))
    d, m, m_d = np.split(scales, 3, axis=-2)
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mins = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape((-1, 8)), mins.reshape((-1, 8))


def _unpack_q4_k(blob: bytes):
    blocks = np.frombuffer(blob, dtype=np.uint8).reshape(-1, _LAYOUT["Q4_K"][0])
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc, mins = _q4_k_scale_min(blocks[:, 4:16])

    scale = d * sc.astype(np.float32)          # (nb, 8)
    bias = -(dmin * mins.astype(np.float32))   # (nb, 8)

    # Low nibbles of a 32-byte chunk are one group, high nibbles the next,
    # which is the order the scales are in.
    qs = blocks[:, 16:].reshape(-1, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    q = (qs & np.uint8(0x0F)).reshape(-1, 8, 32)
    return q, scale, bias


def _nibbles(qs: np.ndarray, per_half: int) -> np.ndarray:
    """The 4-bit halves of a byte run, low half first then the high half.

    GGML puts the first `per_half` weights in the low nibbles and the next
    `per_half` in the high nibbles of the same bytes, so the order is not
    simply "each byte gives two consecutive weights".
    """
    split = qs.reshape(-1, qs.shape[1] // per_half, 1, per_half) >> np.array(
        [0, 4], dtype=np.uint8
    ).reshape(1, 1, 2, 1)
    return (split & np.uint8(0x0F)).reshape(qs.shape[0], -1)


def _unpack_q4_0(blob: bytes):
    blocks = np.frombuffer(blob, dtype=np.uint8).reshape(-1, _LAYOUT["Q4_0"][0])
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    q = _nibbles(blocks[:, 2:], 16)
    return q.reshape(-1, 1, 32), d, -8.0 * d


def _unpack_q4_1(blob: bytes):
    blocks = np.frombuffer(blob, dtype=np.uint8).reshape(-1, _LAYOUT["Q4_1"][0])
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    m = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    q = _nibbles(blocks[:, 4:], 16)
    return q.reshape(-1, 1, 32), d, m


def _fifth_bits(qh: np.ndarray) -> np.ndarray:
    """Q5_0/Q5_1 keep the top bit of all 32 weights in one 32-bit word."""
    word = qh.copy().view(np.uint32).reshape(-1, 1)
    return ((word >> np.arange(32, dtype=np.uint32).reshape(1, 32)) & np.uint32(1)).astype(np.uint8)


def _unpack_q5_0(blob: bytes):
    blocks = np.frombuffer(blob, dtype=np.uint8).reshape(-1, _LAYOUT["Q5_0"][0])
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    q = _nibbles(blocks[:, 6:], 16) | (_fifth_bits(blocks[:, 2:6]) << np.uint8(4))
    return q.reshape(-1, 1, 32), d, -16.0 * d


def _unpack_q5_1(blob: bytes):
    blocks = np.frombuffer(blob, dtype=np.uint8).reshape(-1, _LAYOUT["Q5_1"][0])
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    m = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    q = _nibbles(blocks[:, 8:], 16) | (_fifth_bits(blocks[:, 4:8]) << np.uint8(4))
    return q.reshape(-1, 1, 32), d, m


def _unpack_q5_k(blob: bytes):
    blocks = np.frombuffer(blob, dtype=np.uint8).reshape(-1, _LAYOUT["Q5_K"][0])
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc, mins = _q4_k_scale_min(blocks[:, 4:16])

    scale = d * sc.astype(np.float32)
    bias = -(dmin * mins.astype(np.float32))

    qh = blocks[:, 16:48].reshape(-1, 1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)
    qh = (qh & np.uint8(0x01)).reshape(-1, 8, 32)
    ql = blocks[:, 48:].reshape(-1, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    ql = (ql & np.uint8(0x0F)).reshape(-1, 8, 32)
    return ql | (qh << np.uint8(4)), scale, bias


def _unpack_q8_0(blob: bytes):
    blocks = np.frombuffer(blob, dtype=np.uint8).reshape(-1, _LAYOUT["Q8_0"][0])
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    # GGML stores signed bytes; the affine form wants unsigned, so the shift
    # by 128 moves into the bias rather than touching the values.
    q = (blocks[:, 2:].view(np.int8).astype(np.int16) + 128).astype(np.uint8)
    return q.reshape(-1, 1, 32), d, -128.0 * d


_UNPACK = {
    "Q4_0": _unpack_q4_0,
    "Q4_1": _unpack_q4_1,
    "Q5_0": _unpack_q5_0,
    "Q5_1": _unpack_q5_1,
    "Q8_0": _unpack_q8_0,
    "Q4_K": _unpack_q4_k,
    "Q5_K": _unpack_q5_k,
}


def _pack_words(q: np.ndarray, bits: int) -> np.ndarray:
    """Pack small integers into uint32 words as one little-endian bit stream.

    Value i occupies bits [i*bits, (i+1)*bits) of the stream, which is what
    mlx's kernels read -- checked against what ``mx.quantize`` itself emits,
    for every width it supports. Widths that do not divide 32 (5 and 6, both
    of which GGML uses) straddle word boundaries, so this cannot be done a
    word at a time.
    """
    # Expert weights arrive as one 3-D tensor per projection, so the
    # leading axes are flattened and restored rather than assumed away.
    original = q.shape
    if 32 % bits == 0:
        q = np.ascontiguousarray(q, dtype=np.uint8).reshape(-1, original[-1])
        vals_per_word = 32 // bits
        n_words = q.shape[-1] // vals_per_word
        grouped = q.reshape(q.shape[0], n_words, vals_per_word).astype(np.uint32)
        shifts = (np.arange(vals_per_word, dtype=np.uint32) * bits
                  ).reshape(1, 1, -1)
        words = np.bitwise_or.reduce(grouped << shifts, axis=-1)
        return words.reshape(*original[:-1], -1)

    flat = np.ascontiguousarray(q, dtype=np.uint8).reshape(-1, original[-1], 1)
    stream = np.unpackbits(flat, axis=-1, bitorder="little")[:, :, :bits]
    packed = np.packbits(stream.reshape(stream.shape[0], -1), axis=-1,
                         bitorder="little")
    words = np.ascontiguousarray(packed).view(np.uint32)
    return words.reshape(*original[:-1], -1)


def repack(blob: bytes, dtype: str, shape: tuple):
    """A GGML tensor as (packed, scales, biases, bits) in mlx's affine layout.

    `shape` is the row-major shape of the weight matrix. Returns None when
    the type has no exact affine form.
    """
    if dtype not in _UNPACK:
        return None

    bits = _LAYOUT[dtype][2]
    q, scale, bias = _UNPACK[dtype](blob)

    n_elements = int(np.prod(shape))
    q = q.reshape(-1)[:n_elements].reshape(shape)
    n_groups = shape[-1] // AFFINE_GROUP
    scale = scale.reshape(-1)[: n_elements // AFFINE_GROUP].reshape(*shape[:-1], n_groups)
    bias = bias.reshape(-1)[: n_elements // AFFINE_GROUP].reshape(*shape[:-1], n_groups)

    return _pack_words(q, bits), scale, bias, bits
