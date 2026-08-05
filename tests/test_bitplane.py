"""The bit-plane delta encoding: reversible, selective, and backward compatible.

Bit-plane splitting is chosen per tensor by the same smallest-wins rule as
every other encoding, so the risk is not that it picks badly -- it cannot,
since it only wins by being smaller -- but that it decodes wrongly, or that it
is offered for a dtype whose bytes it cannot reassemble. These check both, and
that packs written before it existed still read.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reminis.diff import (  # noqa: E402
    KNOWN_ENCODINGS,
    _decode_bitplane,
    _decode_tensor,
    _encode_bitplane,
    _encode_delta,
    _merge_planes,
    _split_planes,
)

checks = 0


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        print(f"  FAIL: {label}")
        sys.exit(1)
    print(f"  ok: {label}")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_planes_roundtrip():
    section("Splitting into planes and merging back is the identity")
    rng = np.random.default_rng(0)
    for width in (2, 4):
        for count in (1, 7, 1000):
            data = rng.integers(0, 256, size=count * width, dtype=np.uint8).tobytes()
            planes = _split_planes(data, width)
            check(len(planes) == width, f"width {width}: produced {width} planes")
            check(all(len(p) == count for p in planes),
                  f"width {width}, {count} values: planes are equal length")
            check(_merge_planes(planes, width) == data,
                  f"width {width}, {count} values: merge undoes split")


def test_plane_contents():
    section("Planes deinterleave by byte position, not by host endianness")
    # Two little-endian f16 words: 0x1234 and 0xABCD.
    data = bytes([0x34, 0x12, 0xCD, 0xAB])
    low, high = _split_planes(data, 2)
    check(low == bytes([0x34, 0xCD]), "plane 0 holds the low bytes")
    check(high == bytes([0x12, 0xAB]), "plane 1 holds the high bytes")


def test_encode_decode():
    section("Encoding and decoding a delta round-trips exactly")
    rng = np.random.default_rng(1)
    for dtype in ("F16", "BF16"):
        delta = rng.integers(0, 256, size=4096, dtype=np.uint8)
        payload = _encode_bitplane(delta, dtype)
        check(payload is not None, f"{dtype}: encoder produced a payload")
        back = _decode_bitplane(payload, dtype, "t")
        check(back == delta.tobytes(), f"{dtype}: decode reproduces the delta")


def test_refusals():
    section("The encoder declines what it cannot represent")
    rng = np.random.default_rng(2)
    delta = rng.integers(0, 256, size=1024, dtype=np.uint8)

    check(_encode_bitplane(delta, "F32") is None,
          "declines F32, which has no plane layout")
    check(_encode_bitplane(delta, "Q4_K") is None,
          "declines a quantized dtype")

    odd = rng.integers(0, 256, size=1025, dtype=np.uint8)
    check(_encode_bitplane(odd, "F16") is None,
          "declines a byte count that is not a whole number of values")

    try:
        _decode_bitplane(b"\x00" * 8, "F32", "t")
        check(False, "decoding an F32 bit-plane payload raises")
    except ValueError as exc:
        check("no plane layout" in str(exc),
              "decoding an F32 bit-plane payload raises, and says why")

    try:
        _decode_bitplane(b"\xff\xff\xff\xff", "F16", "t")
        check(False, "a truncated payload raises")
    except ValueError as exc:
        check("Truncated" in str(exc), "a truncated payload raises, and says why")


def test_selection():
    section("Bit-plane is chosen only when it is smaller")
    rng = np.random.default_rng(3)

    # A realistic f16 fine-tune delta: exponent bytes mostly unchanged,
    # mantissa bytes randomised. This is the case the split exists for.
    n = 200_000
    high = np.where(rng.random(n) < 0.85, 0, rng.integers(1, 8, n)).astype(np.uint8)
    low = rng.integers(0, 256, n, dtype=np.uint8)
    b_arr = np.empty(n * 2, dtype=np.uint8)
    b_arr[0::2] = low
    b_arr[1::2] = high
    a_blob = bytes(n * 2)
    b_blob = b_arr.tobytes()

    encoding, payload = _encode_delta(a_blob, b_blob, "F16")
    check(encoding == "bitplane_zstd",
          "a structured f16 delta selects the bit-plane encoding")

    decoded = _decode_tensor(encoding, payload, a_blob, "F16", "t")
    check(decoded == b_blob, "the selected encoding decodes back to the target")

    # Incompressible on both plans: the split adds framing and cannot win.
    noise = rng.integers(0, 256, n * 2, dtype=np.uint8).tobytes()
    encoding, _ = _encode_delta(a_blob, noise, "F16")
    check(encoding != "bitplane_zstd",
          "pure noise does not select bit-plane, which would only add framing")

    # A dtype with no plane layout must be untouched by any of this.
    encoding, payload = _encode_delta(a_blob, b_blob, "Q4_K")
    check(encoding in ("xor_zstd", "replace_zstd"),
          "a quantized dtype keeps an encoding older readers understand")
    check(_decode_tensor(encoding, payload, a_blob, "Q4_K", "t") == b_blob,
          "and still decodes correctly")


def test_backward_compatibility():
    section("Older encodings still read")
    check("bitplane_zstd" in KNOWN_ENCODINGS, "the new encoding is registered")
    for old in ("xor_zstd", "replace_zstd", "xor_zlib", "replace_zlib", "lowrank_zstd"):
        check(old in KNOWN_ENCODINGS, f"'{old}' is still accepted")

    import zlib
    base = bytes(range(256))
    target = bytes((b + 7) % 256 for b in base)
    delta = bytes(a ^ b for a, b in zip(base, target))
    payload = zlib.compress(delta)
    check(_decode_tensor("xor_zlib", payload, base, "F16", "t") == target,
          "a pre-0.3.0 zlib pack decodes with the bit-plane path in place")


def main():
    test_planes_roundtrip()
    test_plane_contents()
    test_encode_decode()
    test_refusals()
    test_selection()
    test_backward_compatibility()
    print("\n" + "=" * 70)
    print(f"ALL BIT-PLANE TESTS PASSED ({checks} checks)")
    print("=" * 70)


if __name__ == "__main__":
    main()
