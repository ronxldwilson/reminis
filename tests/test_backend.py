"""Verify that every backend computes the same thing.

numpy is the reference: its logits are the ones checked against transformers
to five decimal places. A GPU backend earns its place by agreeing with it,
not by being fast, so these checks are about agreement. Speed is measured
separately and is not a test.

Backends that this machine does not have are skipped rather than failed --
cupy needs NVIDIA hardware, mlx needs Apple silicon, and neither is a
prerequisite for reminis working.
"""

import sys
from pathlib import Path

import numpy as np

from reminis.backend import (
    NumpyBackend,
    available_backends,
    report,
    select,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
SMOL = MODELS_DIR / "SmolLM-135M.f16.db"

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"  --  {detail}" if detail else ""))
        FAILURES.append(label)


def others():
    """Every available backend except the numpy reference."""
    return [n for n in available_backends() if n != "numpy"]


def test_selection():
    print("\nSelection")
    print(report().rstrip())
    print()

    check("numpy is always available", NumpyBackend.available())
    check("numpy is always in the list", "numpy" in available_backends())

    # Byte work is zlib-bound and merge measured slower on a GPU, so both
    # must stay on numpy however much hardware is lying around.
    check("byte work stays on numpy", select("bytes").name == "numpy")
    check("merge arithmetic stays on numpy", select("elementwise").name == "numpy")

    check("an unknown backend is refused",
          _raises(lambda: select(requested="tensorflow"), "unknown backend"))

    # An explicit request that cannot be honoured must fail rather than
    # quietly fall back, or a benchmark comparing backends would compare
    # numpy against numpy and report a 1.0x speedup.
    missing = next((n for n in ("mlx", "cupy") if n not in available_backends()), None)
    if missing:
        check(f"requesting unavailable '{missing}' fails loudly",
              _raises(lambda: select(requested=missing), "not available"))
    else:
        print("  skip  every backend is available, nothing to refuse")


def _raises(fn, fragment):
    try:
        fn()
    except ValueError as exc:
        return fragment in str(exc).lower()
    return False


def test_primitives():
    """Each backend's building blocks against numpy's."""
    print("\nPrimitives against the numpy reference")
    ref = NumpyBackend()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    w = rng.standard_normal(64).astype(np.float32)

    for name in others():
        b = select(requested=name)
        gx, gw = b.from_numpy(x), b.from_numpy(w)

        pairs = [
            ("rms_norm", ref.rms_norm(x, w, 1e-5), b.rms_norm(gx, gw, 1e-5)),
            ("silu", ref.silu(x), b.silu(gx)),
            ("softmax", ref.softmax(x), b.softmax(gx)),
        ]
        for op, want, got in pairs:
            got = b.to_numpy(got)
            # float16 compute means these agree to about three decimals,
            # not to the bit -- which is the trade being made deliberately.
            close = np.allclose(want, got, atol=2e-3, rtol=2e-3)
            check(f"{name}: {op} matches numpy",
                  close, f"max diff {np.abs(want - got).max():.2e}")


def test_roundtrip():
    """Bytes in and bytes out must survive every backend unchanged."""
    print("\nStored bytes through each backend")
    from reminis.dtypes import from_float32

    rng = np.random.default_rng(7)
    values = rng.standard_normal(2048).astype(np.float32)

    for name in ["numpy"] + others():
        b = select("elementwise", requested=name)
        for dtype in ("F32", "F16", "BF16"):
            raw = from_float32(values, dtype)
            decoded = b.from_bytes(raw, dtype)
            reencoded = b.to_bytes(decoded, dtype)
            check(f"{name}: {dtype} survives a decode and re-encode",
                  reencoded == raw,
                  f"{len(reencoded)} bytes vs {len(raw)}")


def test_forward_pass_agrees():
    """The whole model, on each backend, against numpy's answer."""
    print("\nA full forward pass on each backend")
    if not SMOL.exists():
        print("  skip  SmolLM-135M.f16.db not present")
        return
    if not others():
        print("  skip  no backend beyond numpy on this machine")
        return

    from reminis.infer import KVCache, Model

    prompt = "The capital of France is Paris, and the capital of Germany is"
    reference = None
    for name in ["numpy"] + others():
        b = select(requested=name)
        model = Model(str(SMOL), backend=b)
        try:
            ids = model.tokenizer.encode(prompt, add_special=False)
            logits = model.forward(ids, KVCache(model.cfg.n_layers, capacity=64,
                                                backend=b), 0)
        finally:
            model.close()

        if reference is None:
            reference = logits
            continue

        # Agreement is judged on the ranking, not the values. Half precision
        # moves a logit by a few hundredths, which never changes which token
        # is chosen but would fail a tight equality check.
        check(f"{name}: picks the same next token",
              int(np.argmax(logits)) == int(np.argmax(reference)))
        corr = float(np.corrcoef(logits, reference)[0, 1])
        check(f"{name}: logits track numpy's (corr > 0.9999)", corr > 0.9999,
              f"corr {corr:.7f}")
        print(f"        (largest logit difference: "
              f"{np.abs(logits - reference).max():.3f}, correlation {corr:.7f})")


def test_generation_agrees():
    """Greedy text must be the same on every backend."""
    print("\nGreedy generation on each backend")
    if not SMOL.exists():
        print("  skip  SmolLM-135M.f16.db not present")
        return
    if not others():
        print("  skip  no backend beyond numpy on this machine")
        return

    from reminis.infer import generate

    baseline = None
    for name in ["numpy"] + others():
        result = generate(str(SMOL), "The capital of France is", max_tokens=12,
                          temperature=0.0, verbose=False, on_token=lambda _: None,
                          backend=name)
        check(f"{name}: reports which backend it used",
              result["backend"] == name, result["backend"])
        if baseline is None:
            baseline = result["completion"]
            print(f"        (numpy said: {baseline!r})")
            continue
        check(f"{name}: generates the same text as numpy",
              result["completion"] == baseline,
              f"got {result['completion']!r}")


def main():
    print("=" * 70)
    print("BACKEND TESTS")
    print("=" * 70)

    test_selection()
    test_primitives()
    test_roundtrip()
    test_forward_pass_agrees()
    test_generation_agrees()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} BACKEND TESTS FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 70)
        sys.exit(1)
    print("ALL BACKEND TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
