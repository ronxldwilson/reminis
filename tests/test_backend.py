"""Verify that every backend computes the same thing.

numpy is the reference: its logits are the ones checked against transformers
to five decimal places. A GPU backend earns its place by agreeing with it,
not by being fast, so these checks are about agreement. Speed is measured
separately and is not a test.

Backends that this machine does not have are skipped rather than failed --
cupy needs NVIDIA hardware, mlx needs Apple silicon, and neither is a
prerequisite for reminis working.
"""

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


def test_attention_features():
    """Attention sinks and sliding windows, backend against backend.

    Both change what a model computes rather than how fast it computes it,
    so a backend that quietly ignores either would still produce fluent
    text. These check that each one alters the result at all, and that the
    array-op fallback and the fused kernel agree about how.
    """
    print("\nAttention sinks and sliding windows")
    if not others():
        print("  skip  no backend beyond numpy on this machine")
        return

    rng = np.random.default_rng(0)
    batch, heads, kv_heads, tokens, keys, dim = 1, 8, 2, 3, 16, 32
    q = rng.standard_normal((batch, heads, tokens, dim)).astype(np.float32)
    k = rng.standard_normal((batch, kv_heads, keys, dim)).astype(np.float32)
    v = rng.standard_normal((batch, kv_heads, keys, dim)).astype(np.float32)
    sink = rng.standard_normal(heads).astype(np.float32)
    scale = 1.0 / np.sqrt(dim)

    rows = np.arange(keys - tokens, keys)[:, None]
    cols = np.arange(keys)[None, :]
    window = (cols <= rows) & (cols > rows - 5)

    def run(name):
        b = select(requested=name)
        args = (b.from_numpy(q), b.from_numpy(k), b.from_numpy(v))
        m = b.xp.array(window)
        sk = b.from_numpy(sink)
        return {
            "plain": b.to_numpy(b.attention(*args, scale)),
            "sinks": b.to_numpy(b.attention(*args, scale, None, sk)),
            "window": b.to_numpy(b.attention(*args, scale, m)),
            "both": b.to_numpy(b.attention(*args, scale, m, sk)),
        }

    reference = run("numpy")
    check("a sink changes the result", not np.allclose(
        reference["plain"], reference["sinks"], atol=1e-4))
    check("a window changes the result", not np.allclose(
        reference["plain"], reference["window"], atol=1e-4))

    for name in others():
        got = run(name)
        for kind in ("plain", "sinks", "window", "both"):
            gap = float(np.abs(got[kind] - reference[kind]).max())
            check(f"{name}: {kind} attention matches numpy",
                  np.allclose(got[kind], reference[kind], atol=3e-3),
                  f"max diff {gap:.2e}")


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
