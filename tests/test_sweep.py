"""The precision sweep: comparison maths, fit prediction, and reference choice.

The risk in a sweep is not that it crashes but that it reports something
confident and wrong -- comparing against a reference that silently changed,
measuring one position and calling it agreement, or counting only half the
resident weights so that packing appears to cost memory. These check the
pieces that would produce a plausible bad table.
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reminis.sweep import (  # noqa: E402
    DEFAULT_PROMPT,
    FIT_SHARE,
    _compare,
    _predicted_bytes,
    _softmax,
    sweep,
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


def test_softmax():
    section("Softmax is stable and normalised")
    x = np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32)
    p = _softmax(x)
    check(np.isfinite(p).all(), "no overflow on large logits")
    check(abs(float(p.sum()) - 1.0) < 1e-6, "rows sum to one")


def test_compare_identical():
    section("Identical logits compare as identical")
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 500)).astype(np.float32)
    stats = _compare(a, a.copy())
    check(stats["top1"] == 1.0, "top-1 agreement is 100%")
    check(stats["top5"] == 1.0, "top-5 overlap is 100%")
    check(abs(stats["kl"]) < 1e-6, "KL divergence is zero")


def test_compare_detects_damage():
    section("Comparison notices a degraded distribution")
    rng = np.random.default_rng(1)
    a = rng.normal(size=(40, 500)).astype(np.float32)

    noisy = a + rng.normal(scale=0.01, size=a.shape).astype(np.float32)
    wrecked = a + rng.normal(scale=5.0, size=a.shape).astype(np.float32)

    mild = _compare(a, noisy)
    bad = _compare(a, wrecked)

    check(mild["top1"] > bad["top1"], "small noise agrees more than large noise")
    check(mild["kl"] < bad["kl"], "KL grows with the damage")
    check(bad["top1"] < 0.5, "heavy noise loses most of the top-1 agreement")

    # A permuted argmax must not pass: this is the failure that would make a
    # broken rung look fine.
    shifted = np.roll(a, 1, axis=-1)
    check(_compare(a, shifted)["top1"] < 0.1,
          "shifting every logit by one position destroys agreement")


def test_compare_uses_every_position():
    section("Agreement is measured over all positions, not just one")
    rng = np.random.default_rng(2)
    a = rng.normal(size=(100, 50)).astype(np.float32)
    b = a.copy()
    # Break exactly a quarter of the positions.
    b[:25] = rng.normal(size=(25, 50)).astype(np.float32)
    stats = _compare(a, b)
    check(0.7 <= stats["top1"] <= 0.8,
          f"a quarter broken gives ~75% agreement (got {stats['top1']:.0%})")


def test_predicted_bytes():
    section("Predicted resident size tracks the width")
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "m.db")
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE tensors (id INTEGER PRIMARY KEY, name TEXT, "
            "shape TEXT, dtype TEXT, dtype_id INTEGER, n_elements INTEGER, "
            "n_bytes INTEGER, data BLOB)"
        )
        conn.execute(
            "INSERT INTO tensors (name, shape, dtype, dtype_id, n_elements, "
            "n_bytes, data) VALUES ('w','[8]','F16',1,1_000_000,0,X'00')"
        )
        conn.commit()
        conn.close()

        f16 = _predicted_bytes(path, None)
        eight = _predicted_bytes(path, 8)
        four = _predicted_bytes(path, 4)

        check(f16 == 2_000_000, "float16 is two bytes a weight")
        check(eight < f16, "8-bit is smaller than float16")
        check(four < eight, "4-bit is smaller than 8-bit")
        # bits + 2 for the group scales, so 4-bit is 6/16 of float16.
        check(four == int(1_000_000 * 6 / 8),
              "the group scale overhead is counted, not ignored")


def test_fit_share():
    section("The fit margin leaves headroom")
    check(0 < FIT_SHARE < 1, "the share is a fraction of the working set")
    check(FIT_SHARE <= 0.9, "it leaves room for activations and the KV cache")


def test_end_to_end():
    section("A real sweep produces a coherent table")
    model = MODELS / "SmolLM-135M.f16.db"
    if not model.exists():
        print(f"  skipped: {model} not present")
        return

    try:
        result = sweep(str(model), [8, 4], verbose=False)
    except Exception as exc:
        if "cannot hold packed" in str(exc):
            print(f"  skipped: {exc}")
            return
        raise

    rows = [r for r in result["rows"] if "skipped" not in r]
    check(len(rows) >= 2, "the reference and at least one rung were measured")

    reference = rows[0]
    check(reference["top1"] == 1.0, "the reference agrees with itself")
    check(reference["share"] == 1.0, "the reference is 100% of itself")

    packed = {r["bits"]: r for r in rows[1:]}
    check(all(r["share"] < 1.0 for r in packed.values()),
          "every packed rung is smaller than the reference")
    check(packed[4]["bytes"] < packed[8]["bytes"],
          "4-bit is resident in less memory than 8-bit")
    check(packed[4]["kl"] > packed[8]["kl"],
          "4-bit diverges further from the reference than 8-bit")
    check(packed[8]["top1"] >= packed[4]["top1"],
          "8-bit agrees at least as often as 4-bit")

    # The bug this guards: counting only the store's cache misses the fused
    # weights the reference holds, making full precision look smaller than
    # a packed rung.
    check(reference["bytes"] > packed[8]["bytes"],
          "the reference is counted in full, including fused weights")


def test_prompt_is_substantial():
    section("The default prompt is long enough to mean something")
    check(len(DEFAULT_PROMPT.split()) > 30,
          "the default prompt is more than a handful of words")


def main():
    test_softmax()
    test_compare_identical()
    test_compare_detects_damage()
    test_compare_uses_every_position()
    test_predicted_bytes()
    test_fit_share()
    test_prompt_is_substantial()
    test_end_to_end()
    print("\n" + "=" * 70)
    print(f"ALL SWEEP TESTS PASSED ({checks} checks)")
    print("=" * 70)


if __name__ == "__main__":
    main()
