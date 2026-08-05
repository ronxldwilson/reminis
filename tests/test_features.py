"""Cover the parts the per-subject suites leave between them.

The other test files each take one subject -- the backends, the repack, the
forward pass -- and check it in isolation. What falls through the gaps is
everything that only exists where two of them meet: packing combined with a
compressed cache, a mixture of experts that is also quantized, streaming a
model whose weights have to be unpacked on every read.

The command line is here too, because none of the other suites touch it. A
flag that silently does nothing, or one that accepts a value it cannot
honour, is invisible to tests that import the library directly.
"""

import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

from reminis.backend import available_backends, select
from reminis.ggml_affine import can_repack, nearest_bits
from reminis.infer import KVCache, Model, _best_group, generate

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
SMOL = MODELS_DIR / "SmolLM-135M.f16.db"
QUANT = MODELS_DIR / "smollm-q4km.db"
MOE = MODELS_DIR / "granite-3.1-1b-a400m-instruct-Q4_K_M.db"

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"  --  {detail}" if detail else ""))
        FAILURES.append(label)


def cli(*args, expect_ok=True):
    """Run the real console script, the way a user would."""
    result = subprocess.run(
        ["uv", "run", "reminis", *args],
        cwd=ROOT, capture_output=True, text=True,
    )
    if expect_ok and result.returncode != 0:
        return result, False
    return result, True


def packs_supported() -> bool:
    return select("inference").can_pack()


# ------------------------------------------------------------ group sizes


def test_group_selection():
    """The largest group that divides the row, and never one that does not.

    A group that does not divide the row length is not merely suboptimal --
    the backend refuses it -- so this is a correctness check wearing a
    performance check's clothes.
    """
    print("\nQuantization group selection")
    check("896 takes the largest group", _best_group(896) == 128)
    check("576 falls to 64, since 128 does not divide it", _best_group(576) == 64)
    check("2048 takes 128", _best_group(2048) == 128)
    check("96 falls to 32", _best_group(96) == 32)
    # A row length divisible by nothing sensible still has to return
    # something the backend accepts.
    check("100 falls back to 32 rather than an unusable size",
          _best_group(100) == 32)
    check("a limit is respected", _best_group(2048, limit=32) == 32)

    if not (packs_supported() and SMOL.exists()):
        print("  skip  no packing backend or model for the live check")
        return

    model = Model(str(SMOL), pack_bits=8)
    try:
        bad = []
        for name in model.store._shapes:
            weight = model.store.get(name)
            group = getattr(weight, "group_size", None)
            if group is None:
                continue
            row = weight.shape[-1]
            if row % group:
                bad.append((name, row, group))
        check("every packed tensor got a group that divides its row",
              not bad, str(bad[:3]))
    finally:
        model.close()


def test_nearest_bits():
    """Types with no affine form map to a width, or decline outright."""
    print("\nFallback widths for non-affine quantizations")
    check("Q6_K keeps six bits", nearest_bits("Q6_K") == 6)
    check("Q2_K widens slightly rather than staying at two",
          nearest_bits("Q2_K") == 3)
    check("the i-quants map to four", nearest_bits("IQ4_XS") == 4)
    check("a float type has no fallback width", nearest_bits("F16") is None)
    check("Q4_K needs no fallback, being exactly affine", can_repack("Q4_K"))
    check("Q6_K is not exactly affine", not can_repack("Q6_K"))


# ------------------------------------------------------------ embeddings


def test_embedding_packing():
    """A packed embedding table must still index to the same rows."""
    print("\nPacked embedding tables")
    if not (packs_supported() and SMOL.exists()):
        print("  skip  no packing backend or model")
        return

    plain = Model(str(SMOL))
    packed = Model(str(SMOL), pack_bits=8)
    try:
        backend = packed.backend
        table_plain = plain.store.get("token_embd.weight")
        table_packed = packed.store.get("token_embd.weight")

        check("the embedding table is packed, not skipped",
              type(table_packed).__name__ == "QuantizedWeight",
              type(table_packed).__name__)

        ids = backend.xp.array(np.array([0, 5, 1000, 40000], dtype=np.int32))
        want = plain.backend.to_numpy(plain.backend.take_rows(table_plain, ids))
        got = backend.to_numpy(backend.take_rows(table_packed, ids))
        check("indexing a packed table gives the same rows",
              got.shape == want.shape
              and np.allclose(got, want, atol=5e-2, rtol=5e-2),
              f"max diff {np.abs(got - want).max():.3e}")
        # Rows must be distinct: a bug that returned row 0 for everything
        # would still pass a closeness check against itself.
        check("different ids give different rows",
              not np.allclose(got[0], got[1], atol=1e-3))
    finally:
        plain.close()
        packed.close()


# ------------------------------------------------- sliding-window attention


def test_sliding_window():
    """Which layers are windowed, and that a window changes the answer."""
    print("\nSliding-window layers")
    if not SMOL.exists():
        print("  skip  SmolLM-135M.f16.db not present")
        return

    model = Model(str(SMOL))
    try:
        check("a model with no window slides on no layer",
              not any(model._layer_slides(i) for i in range(model.cfg.n_layers)))

        # Pattern N means one layer in every N sees everything.
        model.cfg.sliding_window = 8
        model.cfg.swa_pattern = 4
        slides = [model._layer_slides(i) for i in range(8)]
        check("with a pattern of 4, every fourth layer sees everything",
              slides == [True, True, True, False] * 2, str(slides))

        model.cfg.swa_pattern = 0
        check("with no pattern, a stated window applies everywhere",
              all(model._layer_slides(i) for i in range(4)))

        # A window narrower than the context must change the result.
        ids = model.tokenizer.encode(
            "The capital of France is Paris, and the capital of Germany is",
            add_special=False,
        )
        model.cfg.sliding_window = 0
        model._mask_cache = {}
        full = model.forward(ids, KVCache(model.cfg.n_layers, capacity=64,
                                          backend=model.backend), 0)
        model.cfg.sliding_window = 3
        model._mask_cache = {}
        windowed = model.forward(ids, KVCache(model.cfg.n_layers, capacity=64,
                                              backend=model.backend), 0)
        check("a narrow window changes the logits",
              not np.allclose(full, windowed, atol=1e-2),
              f"max diff {np.abs(full - windowed).max():.3e}")
    finally:
        model.cfg.sliding_window = 0
        model.close()


# ------------------------------------------------------ feature combinations


def test_combinations():
    """Features that are only exercised where two of them meet."""
    print("\nCombinations")
    if not SMOL.exists():
        print("  skip  SmolLM-135M.f16.db not present")
        return

    quiet = dict(temperature=0.0, verbose=False, on_token=lambda _: None,
                 max_tokens=8)
    baseline = generate(str(SMOL), "The capital of France is", **quiet)["completion"]

    if packs_supported():
        both = generate(str(SMOL), "The capital of France is",
                        pack_bits=8, kv_bits=8, **quiet)["completion"]
        check("packed weights and a compressed cache together still work",
              "paris" in both.lower(), f"got {both!r}")
        check("...and agree with the plain run", both == baseline,
              f"{both!r} vs {baseline!r}")

    if QUANT.exists() and packs_supported():
        for mode in ("native", "compact", 8):
            text = generate(str(QUANT), "The capital of France is",
                            pack_bits=mode, **quiet)["completion"]
            check(f"a quantized model runs with --pack {mode}",
                  "paris" in text.lower(), f"got {text!r}")

    if MOE.exists() and packs_supported():
        text = generate(str(MOE), "The capital of France is",
                        pack_bits="compact", kv_bits=8, **quiet)["completion"]
        check("a mixture of experts runs packed with a compressed cache",
              "paris" in text.lower(), f"got {text!r}")

    if QUANT.exists():
        # Streaming re-reads and re-unpacks every weight, which is a
        # different code path from the cached one for a quantized model.
        streamed = generate(str(QUANT), "The capital of France is",
                            stream=True, **quiet)
        cached = generate(str(QUANT), "The capital of France is", **quiet)
        check("streaming a quantized model matches reading it once",
              streamed["completion"] == cached["completion"],
              f"{streamed['completion']!r} vs {cached['completion']!r}")


def test_database_untouched():
    """Running a model must not modify the database it came from.

    Packing and compression happen in memory. If any of it leaked back to
    disk, a database would quietly change every time it was used.
    """
    print("\nThe database is read-only in practice")
    if not SMOL.exists():
        print("  skip  SmolLM-135M.f16.db not present")
        return

    def digest():
        conn = sqlite3.connect(str(SMOL))
        sha = hashlib.sha256()
        for name, data in conn.execute("SELECT name, data FROM tensors ORDER BY name"):
            sha.update(name.encode())
            sha.update(data)
        conn.close()
        return sha.hexdigest()

    before = digest()
    generate(str(SMOL), "hello", max_tokens=4, temperature=0.0, verbose=False,
             on_token=lambda _: None,
             pack_bits=8 if packs_supported() else None, kv_bits=8
             if packs_supported() else None)
    check("the weights are byte-identical after a run", digest() == before)


# ------------------------------------------------------------ command line


def test_cli_run():
    print("\nCommand line: run")
    if not SMOL.exists():
        print("  skip  SmolLM-135M.f16.db not present")
        return

    result, ok = cli("run", str(SMOL), "The capital of France is",
                     "-n", "6", "--temp", "0")
    check("reminis run generates text", ok and "Paris" in result.stdout,
          result.stdout[-200:] + result.stderr[-200:])

    # A float model has no quantization blocks to rearrange, so bit-exact
    # packing does nothing -- and must say so rather than pretending.
    result, ok = cli("run", str(SMOL), "hi", "-n", "2", "--temp", "0", "--pack")
    check("--pack on a float model reports that it did nothing",
          ok and "nothing to pack" in result.stdout,
          result.stdout[-200:])

    result, _ = cli("run", str(SMOL), "hi", "-n", "2", "--pack", "7",
                    expect_ok=False)
    check("an unsupported --pack width is refused",
          result.returncode != 0, f"exit {result.returncode}")

    result, _ = cli("run", str(SMOL), "hi", "-n", "2", "--backend", "nonsense",
                    expect_ok=False)
    check("an unknown backend is refused", result.returncode != 0)

    result, _ = cli("run", "no-such-model.db", "hi", expect_ok=False)
    check("a missing database exits non-zero", result.returncode != 0)


def test_cli_other_commands():
    print("\nCommand line: info, merge, view")
    if not SMOL.exists():
        print("  skip  SmolLM-135M.f16.db not present")
        return

    result, ok = cli("info", str(SMOL))
    check("reminis info reports the model", ok and "Tensors" in result.stdout)
    check("reminis info lists the compute backends",
          "backend" in result.stdout.lower(), result.stdout[-200:])

    result, _ = cli("merge", str(SMOL), str(SMOL), "-o", str(SMOL),
                    expect_ok=False)
    check("merging onto one of its own inputs is refused",
          result.returncode != 0 or "also an input" in result.stdout,
          result.stdout[-200:])

    result, _ = cli("merge", str(SMOL), "-o", "/tmp/one.db", expect_ok=False)
    check("merging a single model is refused",
          result.returncode != 0 or "at least two" in result.stdout,
          result.stdout[-200:])

    result, ok = cli("--version")
    check("--version reports a version",
          ok and re.search(r"\d+\.\d+\.\d+", result.stdout), result.stdout)


def test_cli_rejects_registry():
    """Single-model commands must not silently aggregate a registry."""
    print("\nCommand line: registries are not models")
    tmp = ROOT / "tests" / "tmp_features"
    tmp.mkdir(exist_ok=True)
    registry = tmp / "reg.db"
    try:
        if not SMOL.exists():
            print("  skip  SmolLM-135M.f16.db not present")
            return
        conn = sqlite3.connect(str(registry))
        conn.executescript(
            "CREATE TABLE models (name TEXT PRIMARY KEY, kind TEXT);"
            "CREATE TABLE tensors (id INTEGER PRIMARY KEY, name TEXT, "
            "shape TEXT, dtype TEXT, dtype_id INT, n_elements INT, "
            "n_bytes INT, data BLOB);"
        )
        conn.commit()
        conn.close()

        for command in ("info", "view"):
            result, _ = cli(command, str(registry), expect_ok=False)
            check(f"reminis {command} refuses a registry",
                  result.returncode != 0 and "registry" in result.stdout.lower(),
                  result.stdout[-160:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


GPTOSS = MODELS_DIR / "gptoss20b.db"


def test_yarn():
    """YaRN must stretch the slow dimensions and leave the fast ones alone.

    That asymmetry is the whole idea: interpolating every frequency equally
    preserves long-range structure and destroys the detail nearby tokens
    depend on. A wrong implementation still produces a plausible-looking
    array of numbers, so the shape is what gets checked.
    """
    print("\nYaRN rotary scaling")
    from reminis.infer import _yarn_periods

    dim, base, scale = 64, 150000.0, 32.0
    periods = _yarn_periods(dim, base, scale, 4096, 32.0, 1.0)
    plain = base ** (np.arange(dim // 2) * 2.0 / dim)
    ratio = periods / plain

    check("the fastest dimensions are left untouched",
          abs(ratio[0] - 1.0) < 0.01, f"ratio {ratio[0]:.4f}")
    check("the slowest dimensions are stretched by the full factor",
          abs(ratio[-1] - scale) < 0.01, f"ratio {ratio[-1]:.4f}")
    # The periods are float32, so the flat stretches wobble by an ulp --
    # at a ratio of 32 that is around 4e-06, which is not a fall in the ramp.
    wobble = np.finfo(np.float32).eps * scale * 4
    check("the ramp between them never decreases",
          bool(np.all(np.diff(ratio) >= -wobble)),
          f"largest fall {np.diff(ratio).min():.2e} against tolerance {wobble:.2e}")
    check("nothing is stretched beyond the factor",
          bool(np.all(ratio <= scale + 1e-3)))

    # With no original context recorded there is no ramp to compute, and
    # every frequency is interpolated equally.
    plainly = _yarn_periods(dim, base, scale, 0, 32.0, 1.0)
    check("without an original context length it is plain interpolation",
          np.allclose(plainly / plain, scale, rtol=1e-5))


def test_gptoss_activation():
    """gpt-oss's gated activation is not SwiGLU, and must not be treated as it."""
    print("\ngpt-oss gated activation")
    if not SMOL.exists():
        print("  skip  need a model to borrow a backend from")
        return

    model = Model(str(SMOL))
    try:
        b = model.backend
        gate = b.from_numpy(np.array([[-2.0, 0.5, 9.0, 3.0]], dtype=np.float32))
        up = b.from_numpy(np.array([[1.0, -9.0, 2.0, 0.0]], dtype=np.float32))

        standard = b.to_numpy(model._glu(gate, up))
        want_standard = b.to_numpy(b.silu(gate) * up)
        check("a normal model gets plain SwiGLU",
              np.allclose(standard, want_standard, atol=1e-3))

        model.cfg.glu_alpha, model.cfg.glu_limit = 1.702, 7.0
        model.cfg.glu_up_offset = 1.0
        got = b.to_numpy(model._glu(gate, up))

        g = np.minimum(np.array([[-2.0, 0.5, 9.0, 3.0]]), 7.0)
        u = np.clip(np.array([[1.0, -9.0, 2.0, 0.0]]), -7.0, 7.0)
        want = (u + 1.0) * (g / (1.0 + np.exp(-1.702 * g)))
        check("gpt-oss clamps, scales the sigmoid, and offsets up by one",
              np.allclose(got, want, atol=2e-2),
              f"got {got.round(3)}, want {want.round(3)}")
        check("...and that is not what SwiGLU gives",
              not np.allclose(got, want_standard, atol=1e-2))
    finally:
        model.cfg.glu_limit = 0.0
        model.close()


def test_mxfp4_on_device():
    """Unpacking MXFP4 on the GPU must match unpacking it in numpy exactly."""
    print("\nMXFP4 unpacked on the device")
    if not GPTOSS.exists():
        print("  skip  no MXFP4 model present")
        return
    backend = select("inference")
    if not hasattr(backend, "dequantize_mxfp4"):
        print(f"  skip  {backend.name} has no device-side MXFP4")
        return

    import ast as _ast

    from reminis.dtypes import to_float32_any

    conn = sqlite3.connect(str(GPTOSS))
    row = conn.execute(
        "SELECT id, dtype, shape, n_bytes FROM tensors "
        "WHERE dtype = 'MXFP4' LIMIT 1"
    ).fetchone()
    if row is None:
        print("  skip  no MXFP4 tensors")
        conn.close()
        return
    rowid, dtype, shape, n_bytes = row
    dims = tuple(_ast.literal_eval(shape))[::-1]
    stride = n_bytes // dims[0]
    handle = conn.blobopen("tensors", "data", rowid, readonly=True)
    raw = handle[:stride]
    handle.close()
    conn.close()

    want = to_float32_any(raw, dtype).reshape(dims[1:])
    got = backend.to_numpy(backend.dequantize_mxfp4(raw, dims[1:]))
    check("device-side MXFP4 is bit-identical to the reference",
          np.array_equal(got, want),
          f"max diff {np.abs(got - want).max():.3e}")
    check("the values look like weights, not noise",
          abs(float(want.mean())) < 0.01 and 0.001 < float(want.std()) < 1.0,
          f"mean {want.mean():.5f} std {want.std():.5f}")


def test_experts_on_demand():
    """Loading experts by routing must give what loading them all gives."""
    print("\nExperts loaded on demand")
    if not MOE.exists():
        print("  skip  no mixture-of-experts model present")
        return
    store_capable = Model(str(MOE))
    capable = store_capable.store.can_stream_experts()
    store_capable.close()
    if not capable:
        print("  skip  incremental blob reads need Python 3.11+")
        return

    texts = {}
    for cache in (0, 64):
        model = Model(str(MOE), expert_cache=cache)
        try:
            ids = model.tokenizer.encode("The capital of France is",
                                         add_special=False)
            kv = KVCache(model.cfg.n_layers, capacity=64, backend=model.backend)
            logits = model.forward(ids, kv, 0)
            out = []
            for _ in range(8):
                token = int(np.argmax(logits))
                out.append(token)
                logits = model.forward([token], kv, kv.length)
            texts[cache] = model.tokenizer.decode(out)
            if cache:
                check("fetching experts by routing actually happened",
                      model.store._expert_misses > 0,
                      f"{model.store._expert_misses} fetches")
                check("the cache is used, not merely filled",
                      model.store._expert_hits > 0,
                      f"{model.store._expert_hits} hits")
        finally:
            model.close()

    check("on-demand experts give the same text as resident ones",
          texts[0] == texts[64], f"{texts[0]!r} vs {texts[64]!r}")


def main():
    print("=" * 70)
    print("FEATURE AND CLI TESTS")
    print("=" * 70)

    test_group_selection()
    test_yarn()
    test_gptoss_activation()
    test_mxfp4_on_device()
    test_experts_on_demand()
    test_nearest_bits()
    test_embedding_packing()
    test_sliding_window()
    test_combinations()
    test_database_untouched()
    test_cli_run()
    test_cli_other_commands()
    test_cli_rejects_registry()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FEATURE TESTS FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 70)
        sys.exit(1)
    print("ALL FEATURE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
