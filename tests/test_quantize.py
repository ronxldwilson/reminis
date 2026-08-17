"""Writing Q8_0 and Q4_0, checked against the gguf package's dequantizer.

A quantizer that is subtly wrong still produces a file that loads and still
produces text, so agreeing with our own dequantizer proves nothing. Every
block layout here is checked against `gguf.quants.dequantize`, which is the
implementation of record, and the error is required to land inside what the
format can represent rather than merely being finite.
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reminis.quantize import (  # noqa: E402
    BITS_TO_FORMAT,
    BLOCK,
    FORMATS,
    _eligible,
    quantize_model,
    quantize_q4_0,
    quantize_q8_0,
)

MODELS = Path(__file__).resolve().parents[1] / "models"
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


def _reference(blob, fmt):
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize
    t = getattr(GGMLQuantizationType, fmt)
    return dequantize(np.frombuffer(blob, dtype=np.uint8), t).astype(np.float32).reshape(-1)


def test_block_sizes():
    section("Blocks are the size the format defines")
    rng = np.random.default_rng(0)
    x = rng.normal(size=BLOCK * 100).astype(np.float32)
    check(len(quantize_q8_0(x)) == 100 * FORMATS["Q8_0"][0],
          "Q8_0 is 34 bytes per 32 weights")
    check(len(quantize_q4_0(x)) == 100 * FORMATS["Q4_0"][0],
          "Q4_0 is 18 bytes per 32 weights")


def test_against_reference():
    section("The gguf package reads back what we wrote")
    rng = np.random.default_rng(1)
    cases = {
        "normal weights": rng.normal(0, 0.02, BLOCK * 300).astype(np.float32),
        "all negative": -np.abs(rng.normal(0, 1, BLOCK * 300)).astype(np.float32),
        "all positive": np.abs(rng.normal(0, 1, BLOCK * 300)).astype(np.float32),
        "wide dynamic range": (rng.normal(0, 1, BLOCK * 300)
                               * 10.0 ** rng.integers(-3, 3, BLOCK * 300)).astype(np.float32),
    }
    # What each format can represent, relative to the block's own extreme.
    #
    # Q8_0 is symmetric, so the bound is half a step: 1/127/2.
    #
    # Q4_0 is not. Its codes span -8..+7 steps and the scale is pinned to the
    # block's extreme, so a weight of equal magnitude and opposite sign clips
    # at 7/8 of where it should be -- an error of one whole step, 1/8, not
    # half of one. That asymmetry is the format, not the quantizer, and a
    # tolerance of half a step here would be demanding better than Q4_0 can do.
    tolerance = {"Q8_0": 1 / 127 / 2 * 1.2, "Q4_0": 1 / 8 * 1.05}

    for fmt, fn in (("Q8_0", quantize_q8_0), ("Q4_0", quantize_q4_0)):
        for label, x in cases.items():
            back = _reference(fn(x), fmt)
            check(np.isfinite(back).all(), f"{fmt}, {label}: all finite")
            check(len(back) == len(x), f"{fmt}, {label}: length preserved")
            # Error is judged per block against that block's own scale, since
            # a per-block format cannot be held to a global one.
            err = np.abs(back - x).reshape(-1, BLOCK).max(axis=1)
            scale = np.abs(x).reshape(-1, BLOCK).max(axis=1)
            rel = err / np.where(scale > 0, scale, 1.0)
            check(rel.max() <= tolerance[fmt],
                  f"{fmt}, {label}: worst block error within the format's grid "
                  f"({rel.max():.4f} <= {tolerance[fmt]:.4f})")


def test_sign_of_the_extreme():
    section("Q4_0 keeps the sign of the block's largest value")
    # A block whose extreme is negative. Taking an unsigned max here would
    # flip every weight in the block, which still round-trips to plausible
    # magnitudes and is why this needs its own check.
    x = np.array([-1.0] + [0.1] * (BLOCK - 1), dtype=np.float32)
    back = _reference(quantize_q4_0(x), "Q4_0")
    check(back[0] < 0, "the negative extreme stays negative")
    check(np.sign(back[1]) == np.sign(x[1]), "the other weights keep their sign")
    check(abs(back[0] - x[0]) < 0.15, "the extreme is represented closely")


def test_zero_blocks():
    section("An all-zero block does not become NaN")
    x = np.zeros(BLOCK * 4, dtype=np.float32)
    for fmt, fn in (("Q8_0", quantize_q8_0), ("Q4_0", quantize_q4_0)):
        back = _reference(fn(x), fmt)
        check(np.isfinite(back).all(), f"{fmt}: zero block stays finite")
        check(np.all(back == 0), f"{fmt}: zero block stays zero")


def test_eligibility():
    section("Only tensors that can be quantized are")
    check(_eligible([256, 128], "F16"), "a 2-D float tensor is eligible")
    check(_eligible([256, 128], "F32"), "F32 is eligible")
    check(not _eligible([128], "F16"), "a 1-D tensor is not")
    check(not _eligible([100, 128], "F16"),
          "a row length that is not a multiple of 32 is not")
    check(not _eligible([256, 128], "Q4_K"),
          "an already-quantized tensor is not, since that would round twice")


def test_widths():
    section("Only the widths that map to a format are offered")
    check(BITS_TO_FORMAT[8] == "Q8_0", "8 bits is Q8_0")
    check(BITS_TO_FORMAT[4] == "Q4_0", "4 bits is Q4_0")
    check(6 not in BITS_TO_FORMAT,
          "6 is not offered, since it would need a k-quant")


def test_refusals():
    section("Refusals")
    with tempfile.TemporaryDirectory() as tmp:
        src = str(Path(tmp) / "m.db")
        conn = sqlite3.connect(src)
        conn.executescript(
            "CREATE TABLE model_meta (key TEXT PRIMARY KEY, value TEXT, type TEXT);"
            "CREATE TABLE tensors (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT UNIQUE, shape TEXT, dtype TEXT, dtype_id INTEGER, "
            "n_elements INTEGER, n_bytes INTEGER, data BLOB);"
        )
        conn.commit()
        conn.close()

        try:
            quantize_model(src, src, bits=4, verbose=False)
            check(False, "quantizing over the input is refused")
        except ValueError as exc:
            check("over itself" in str(exc),
                  "quantizing over the input is refused, and says why")

        try:
            quantize_model(src, str(Path(tmp) / "o.db"), bits=6, verbose=False)
            check(False, "an unsupported width is refused")
        except ValueError as exc:
            check("k-quant" in str(exc),
                  "an unsupported width is refused, naming what it would need")

        try:
            quantize_model(str(Path(tmp) / "nope.db"), str(Path(tmp) / "o2.db"),
                           bits=4, verbose=False)
            check(False, "a missing database is refused")
        except FileNotFoundError:
            check(True, "a missing database is refused")


def test_real_model():
    section("A real model quantizes to the expected size and still runs")
    model = MODELS / "SmolLM-135M.f16.db"
    if not model.exists():
        print(f"  skipped: {model} not present")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "q4.db")
        stats = quantize_model(str(model), out, bits=4, verbose=False)

        check(stats["quantized"] > 0, "some tensors were quantized")
        check(stats["copied"] > 0,
              "norms and 1-D tensors were copied through, not quantized")
        share = stats["new_bytes"] / stats["raw_bytes"]
        check(0.25 < share < 0.32,
              f"Q4_0 lands near 28% of float16 (got {100*share:.1f}%)")

        # Every tensor survived, and the metadata came with it.
        a = sqlite3.connect(str(model))
        b = sqlite3.connect(out)
        n_a = a.execute("SELECT COUNT(*) FROM tensors").fetchone()[0]
        n_b = b.execute("SELECT COUNT(*) FROM tensors").fetchone()[0]
        check(n_a == n_b, "every tensor is present in the output")
        m_a = a.execute("SELECT COUNT(*) FROM model_meta").fetchone()[0]
        m_b = b.execute("SELECT COUNT(*) FROM model_meta").fetchone()[0]
        check(m_a == m_b, "the metadata was carried across")

        # Shapes and element counts must be untouched; only the encoding moved.
        mismatched = 0
        for name, shape, n_elements in a.execute(
                "SELECT name, shape, n_elements FROM tensors"):
            row = b.execute(
                "SELECT shape, n_elements FROM tensors WHERE name = ?",
                (name,)).fetchone()
            if row is None or row[0] != shape or row[1] != n_elements:
                mismatched += 1
        check(mismatched == 0, "shapes and element counts are unchanged")
        a.close()
        b.close()

        try:
            from reminis.infer import generate
        except ImportError:
            print("  skipped: inference unavailable")
            return
        result = generate(out, "The capital of France is", max_tokens=8,
                          temperature=0.0, verbose=False, stream=False)
        text = result.get("completion", "")
        check(len(text.strip()) > 0, "the quantized model generates text")
        check(all(ord(c) < 0x3000 for c in text),
              "the text is not the mojibake a broken block layout produces")
        print(f"      generated: {text!r}")
