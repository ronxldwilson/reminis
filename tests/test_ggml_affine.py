"""Verify that repacking GGML blocks changes no weight at all.

Several GGML quantizations are already affine within each group of 32, which
is the form mlx's quantized matmul wants, so they can be moved into its
layout by shuffling bits rather than by decoding and re-encoding. The claim
that makes it worth doing is *bit-exactness*: unlike re-quantizing, no weight
is rounded a second time.

That claim is the whole test. A repack that is off by one bit somewhere still
produces a model that generates fluent text, so nothing short of comparing
every value against the reference dequantization would catch it. The
reference is the `gguf` package, which is the implementation of record for
the format.
"""

import ast
import sqlite3
import sys
from pathlib import Path

import numpy as np

from reminis.dtypes import to_float32_any
from reminis.ggml_affine import _LAYOUT, _pack_words, can_repack, repack

MODELS_DIR = Path(__file__).parent.parent / "models"
FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"  --  {detail}" if detail else ""))
        FAILURES.append(label)


def have_mlx():
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except Exception:
        return False


def test_bit_packing():
    """Our packing must be byte-identical to what mlx itself emits.

    mlx reads the quantized codes as one little-endian bit stream, and the
    widths GGML uses include 5 and 6, which straddle 32-bit word boundaries.
    Rather than trust a reading of the layout, this recovers the codes from
    mlx's own output and re-packs them.
    """
    print("\nBit packing against mlx's own layout")
    if not have_mlx():
        print("  skip  mlx is not available")
        return
    import mlx.core as mx

    rng = np.random.default_rng(0)
    for bits in (2, 3, 4, 5, 6, 8):
        w = mx.array(rng.standard_normal((4, 256)).astype(np.float32))
        q, s, b = mx.quantize(w, group_size=32, bits=bits)
        mx.eval(q, s, b)
        back = np.array(mx.dequantize(q, s, b, group_size=32, bits=bits))
        scale = np.repeat(np.array(s), 32, axis=1)
        bias = np.repeat(np.array(b), 32, axis=1)
        codes = np.rint((back - bias) / scale).astype(np.uint8)
        check(f"{bits}-bit packing matches mlx",
              np.array_equal(_pack_words(codes, bits), np.array(q)))


def test_repack_is_exact():
    """Every repacked weight must equal what gguf's dequantization gives."""
    print("\nRepacked blocks against gguf's dequantization")
    if not have_mlx():
        print("  skip  mlx is not available")
        return
    import mlx.core as mx

    databases = sorted(MODELS_DIR.glob("*.db"))
    if not databases:
        print("  skip  no databases to read")
        return

    seen = {}
    for path in databases:
        try:
            conn = sqlite3.connect(str(path))
            rows = conn.execute("SELECT name, dtype, shape, data FROM tensors")
        except sqlite3.DatabaseError:
            continue
        for name, dtype, shape, blob in rows:
            if dtype in seen or not can_repack(dtype):
                continue
            dims = tuple(ast.literal_eval(shape))[::-1]
            if len(dims) != 2 or dims[-1] % 32:
                continue
            reference = to_float32_any(blob, dtype).reshape(dims)
            words, scales, biases, bits = repack(blob, dtype, dims)
            back = np.array(mx.dequantize(
                mx.array(words), mx.array(scales), mx.array(biases),
                group_size=32, bits=bits,
            ))
            seen[dtype] = (np.array_equal(back, reference),
                           float(np.abs(back - reference).max()), dims)
        conn.close()

    if not seen:
        print("  skip  no repackable quantized tensors in the local models")
        return

    for dtype, (exact, worst, dims) in sorted(seen.items()):
        check(f"{dtype} repacks bit-exactly ({dims[0]}x{dims[1]})",
              exact, f"largest difference {worst:.3e}")
    print(f"        (types checked: {', '.join(sorted(seen))})")

    untested = sorted(set(_LAYOUT) - set(seen))
    if untested:
        print(f"        (no local tensors to check: {', '.join(untested)})")


def test_refusals():
    """Types with no affine form must decline rather than approximate."""
    print("\nTypes that cannot be repacked")
    # 16-element sub-blocks -- mlx does groups of 32, 64 or 128.
    for dtype in ("Q6_K", "Q2_K", "Q3_K"):
        check(f"{dtype} declines (its sub-blocks are 16 weights)",
              not can_repack(dtype))
    # Codebook types are lookups, not a scale times an integer.
    for dtype in ("IQ3_M", "IQ4_XS", "IQ2_XS"):
        check(f"{dtype} declines (codebook, not affine)", not can_repack(dtype))
    check("a float type declines", not can_repack("F16"))
    check("repack returns None rather than raising",
          repack(b"\x00" * 64, "Q6_K", (2, 32)) is None)


def main():
    print("=" * 70)
    print("GGML AFFINE REPACK TESTS")
    print("=" * 70)

    test_bit_packing()
    test_repack_is_exact()
    test_refusals()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} REPACK TESTS FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 70)
        sys.exit(1)
    print("ALL REPACK TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
