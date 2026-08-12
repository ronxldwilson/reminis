"""Pre-release gate: fast tests that must pass before any version ships.

Runs in under two minutes against SmolLM-135M.f16 (258 MB). Covers every
subsystem that has been burned by a release: conversion, diff, apply,
quantize, merge, registry, viewer, and the bit-plane encoding.

    uv run python tests/test_prerelease.py

The full test suite (`tests/test_roundtrip.py`, `test_infer.py`, etc.)
round-trips every model in `models/` and trains real networks; that is a
thorough check but takes hours. This script is the fast alternative that
catches the common regressions.
"""

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODELS = Path(__file__).resolve().parents[1] / "models"
SMALL_DB = MODELS / "SmolLM-135M.f16.db"
SMALL_GGUF = MODELS / "SmolLM-135M.f16.gguf"

passed = 0
failed = 0
skipped = 0


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def ok(label):
    global passed
    passed += 1
    print(f"  ok: {label}")


def fail(label):
    global failed
    failed += 1
    print(f"  FAIL: {label}")


def skip(label):
    global skipped
    skipped += 1
    print(f"  skip: {label}")


def check(condition, label):
    if condition:
        ok(label)
    else:
        fail(label)


# ── 1. Bit-plane encoding (no model needed) ─────────────────────────────

def test_bitplane():
    section("Bit-plane encoding")
    from reminis.diff import (
        _decode_bitplane, _decode_tensor, _encode_bitplane, _encode_delta,
        _merge_planes, _split_planes,
    )

    rng = np.random.default_rng(0)
    for width in (2, 4):
        data = rng.integers(0, 256, size=500 * width, dtype=np.uint8).tobytes()
        planes = _split_planes(data, width)
        check(_merge_planes(planes, width) == data,
              f"split/merge round-trip at width {width}")

    for dtype in ("F16", "BF16"):
        delta = rng.integers(0, 256, size=4096, dtype=np.uint8)
        payload = _encode_bitplane(delta, dtype)
        check(payload is not None, f"{dtype} encoder produces output")
        back = _decode_bitplane(payload, dtype, "t")
        check(back == delta.tobytes(), f"{dtype} encode/decode round-trip")

    check(_encode_bitplane(rng.integers(0, 256, 1024, dtype=np.uint8), "F32") is None,
          "F32 is declined")
    check(_encode_bitplane(rng.integers(0, 256, 1024, dtype=np.uint8), "Q4_K") is None,
          "quantized dtype is declined")


# ── 2. Quantizer block layout (no model needed) ─────────────────────────

def test_quantize_blocks():
    section("Quantizer block layout")
    from reminis.quantize import BLOCK, FORMATS, quantize_q4_0, quantize_q8_0

    rng = np.random.default_rng(1)
    x = rng.normal(size=BLOCK * 100).astype(np.float32)

    check(len(quantize_q8_0(x)) == 100 * FORMATS["Q8_0"][0],
          "Q8_0 block size correct")
    check(len(quantize_q4_0(x)) == 100 * FORMATS["Q4_0"][0],
          "Q4_0 block size correct")

    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    for fmt, fn in (("Q8_0", quantize_q8_0), ("Q4_0", quantize_q4_0)):
        blob = fn(x)
        t = getattr(GGMLQuantizationType, fmt)
        back = dequantize(np.frombuffer(blob, dtype=np.uint8), t).astype(np.float32).reshape(-1)
        check(np.isfinite(back).all(), f"{fmt} all finite")
        check(len(back) == len(x), f"{fmt} length preserved")

    zeros = np.zeros(BLOCK * 4, dtype=np.float32)
    for fmt, fn in (("Q8_0", quantize_q8_0), ("Q4_0", quantize_q4_0)):
        t = getattr(GGMLQuantizationType, fmt)
        back = dequantize(np.frombuffer(fn(zeros), dtype=np.uint8), t).astype(np.float32).reshape(-1)
        check(np.all(back == 0), f"{fmt} zero block stays zero")


# ── 3. Viewer (no model needed) ─────────────────────────────────────────

def test_viewer():
    section("Viewer")
    from reminis.converter import SCHEMA

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "m.db")
        conn = sqlite3.connect(db)
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT INTO model_meta (key, value, type) VALUES (?, ?, ?)",
            ("general.architecture", "llama", "string"),
        )
        data = np.zeros(128, dtype=np.float16).tobytes()
        conn.execute(
            "INSERT INTO tensors (name, shape, dtype, dtype_id, n_elements, n_bytes, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("test.weight", "[128]", "F16", 1, 128, len(data), data),
        )
        conn.commit()
        conn.close()

        from reminis.viewer import generate_viewer
        html = generate_viewer(db, str(Path(tmp) / "out.html"))
        check(Path(html).exists(), "viewer generates HTML")
        check(Path(html).stat().st_size > 1000, "HTML is non-trivial")

    test_viewer_stats_are_the_stored_values()


def test_viewer_stats_are_the_stored_values():
    """The numbers the viewer reports must be the numbers in the file.

    Through 0.32.1 this decoded BF16 by reading its bytes as float16. The two
    are both 16 bits, so nothing raised -- a stored 100.0 was reported as 3.39,
    and every mean, min, max and heatmap cell of every BF16 model was wrong.
    The check above passed throughout: it asserted the HTML existed.

    Widening to float32 loses nothing for any of these dtypes, so the values
    are asserted exactly rather than within a tolerance.
    """
    from reminis.dtypes import from_float32
    from reminis.viewer import _compute_tensor_stats, _sample_heatmap

    # Chosen so that misreading any one of these dtypes as another shifts the
    # result well outside rounding: 100.0 read as float16 bits comes back 3.39.
    values = np.array([1.5, -2.25, 100.0, 0.0], dtype=np.float32)

    for dtype in ("F32", "F16", "BF16", "F64"):
        blob = from_float32(values, dtype)
        stats = _compute_tensor_stats(blob, dtype, len(values))

        check("error" not in stats and "quantized" not in stats,
              f"{dtype}: viewer decodes it as floats")
        check(stats.get("min") == -2.25, f"{dtype}: min is -2.25, got {stats.get('min')}")
        check(stats.get("max") == 100.0, f"{dtype}: max is 100.0, got {stats.get('max')}")
        check(stats.get("mean") == 24.8125, f"{dtype}: mean is 24.8125, got {stats.get('mean')}")
        check(stats.get("zeros_pct") == 25.0, f"{dtype}: one value in four is zero")
        check(stats.get("n_elements") == 4, f"{dtype}: counts 4 elements")

        # The heatmap decodes separately, so it needs its own assertion. It is
        # normalised by the largest magnitude, which is the 100.0.
        cells = _sample_heatmap(blob, dtype, [2, 2])
        check(cells == [[0.015, -0.023], [1.0, 0.0]],
              f"{dtype}: heatmap is the stored values, got {cells}")

    # A quantized tensor still takes the shortcut rather than being decoded.
    q = _compute_tensor_stats(b"\x00" * 34, "Q4_K", 256)
    check(q.get("quantized") is True, "Q4_K reports as quantized, not as floats")
    check(_sample_heatmap(b"\x00" * 34, "Q4_K", [16, 16]) is None,
          "Q4_K draws no heatmap")


# ── 4. GGUF round-trip (needs SmolLM GGUF) ──────────────────────────────

def test_gguf_roundtrip():
    section("GGUF round-trip")
    if not SMALL_GGUF.exists():
        skip(f"{SMALL_GGUF.name} not present")
        return

    import hashlib

    from gguf.gguf_reader import GGUFReader

    from reminis.converter import (
        _extract_field_value, gguf_to_sqlite, sqlite_to_gguf,
    )

    # Written by the reader on the way in and not part of the file itself.
    READER_ONLY = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}

    def tensor_hashes(path):
        reader = GGUFReader(str(path), mode="r")
        return {t.name: hashlib.sha256(t.data.tobytes()).hexdigest()
                for t in reader.tensors}

    def metadata(path):
        reader = GGUFReader(str(path), mode="r")
        return {
            key: (str(_extract_field_value(field)), list(field.types))
            for key, field in reader.fields.items()
            if key not in READER_ONLY
        }

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "m.db")
        rt = str(Path(tmp) / "m.gguf")

        orig = tensor_hashes(SMALL_GGUF)
        orig_meta = metadata(SMALL_GGUF)
        gguf_to_sqlite(str(SMALL_GGUF), db, verbose=False)
        sqlite_to_gguf(db, rt, verbose=False)
        back = tensor_hashes(Path(rt))
        back_meta = metadata(Path(rt))

        mismatches = [n for n in orig if orig.get(n) != back.get(n)]
        check(len(mismatches) == 0,
              f"tensors survive the round-trip ({len(orig)}, 0 mismatches)")

        # Tensors alone are not enough to call a round-trip lossless. Export
        # dropped every array field through 0.31.1, which took the tokenizer
        # with it: the weights matched, and llama.cpp would not load the file
        # because it could not find the merges.
        dropped = sorted(set(orig_meta) - set(back_meta))
        check(not dropped,
              f"no metadata field is dropped ({len(orig_meta)} fields)"
              + (f" -- lost {dropped}" if dropped else ""))

        changed = sorted(k for k in set(orig_meta) & set(back_meta)
                         if orig_meta[k] != back_meta[k])
        check(not changed,
              "metadata values and types are unchanged"
              + (f" -- differ: {changed[:5]}" if changed else ""))

        for needed in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
            if needed in orig_meta:
                check(needed in back_meta, f"{needed} survives export")

        # The strongest statement available: the file is the file.
        check(Path(rt).read_bytes() == Path(SMALL_GGUF).read_bytes(),
              "exported file is byte-identical to the original")


# ── 5. Diff + apply (needs SmolLM DB) ───────────────────────────────────

def test_diff_apply():
    section("Diff and apply")
    if not SMALL_DB.exists():
        skip(f"{SMALL_DB.name} not present")
        return

    from reminis.diff import _weights_hash, apply_delta, diff_models

    with tempfile.TemporaryDirectory() as tmp:
        target = str(Path(tmp) / "target.db")
        delta = str(Path(tmp) / "delta.db")
        result = str(Path(tmp) / "result.db")

        shutil.copyfile(SMALL_DB, target)

        conn = sqlite3.connect(target)
        touched = [r[0] for r in conn.execute(
            "SELECT name FROM tensors WHERE dtype IN ('F16','F32') LIMIT 3"
        ).fetchall()]
        rng = np.random.default_rng(0)
        for name in touched:
            row = conn.execute("SELECT dtype, data FROM tensors WHERE name = ?", (name,)).fetchone()
            dtype, blob = row
            np_dtype = np.float32 if dtype == "F32" else np.float16
            arr = np.frombuffer(blob, dtype=np_dtype).astype(np.float32)
            noise = rng.normal(0, 0.01, size=arr.shape).astype(np.float32)
            new = (arr + noise).astype(np_dtype)
            conn.execute("UPDATE tensors SET data = ? WHERE name = ?", (new.tobytes(), name))
        conn.commit()
        conn.close()

        summary = diff_models(str(SMALL_DB), target, delta, verbose=False)
        check(summary["changed"] == len(touched),
              f"diff detected {len(touched)} changed tensors")

        apply_delta(str(SMALL_DB), delta, result, verify=True, verbose=False)

        conn_t = sqlite3.connect(target)
        conn_r = sqlite3.connect(result)
        check(_weights_hash(conn_t) == _weights_hash(conn_r),
              "apply produces byte-identical result")
        conn_t.close()
        conn_r.close()

    # Wrong base is rejected
    with tempfile.TemporaryDirectory() as tmp:
        target = str(Path(tmp) / "target.db")
        delta = str(Path(tmp) / "delta.db")
        wrong = str(Path(tmp) / "wrong.db")

        shutil.copyfile(SMALL_DB, target)
        conn = sqlite3.connect(target)
        name = conn.execute("SELECT name FROM tensors WHERE dtype='F16' LIMIT 1").fetchone()[0]
        row = conn.execute("SELECT data FROM tensors WHERE name = ?", (name,)).fetchone()
        arr = np.frombuffer(row[0], dtype=np.float16).copy()
        arr[0] += 1.0
        conn.execute("UPDATE tensors SET data = ? WHERE name = ?", (arr.tobytes(), name))
        conn.commit()
        conn.close()

        diff_models(str(SMALL_DB), target, delta, verbose=False)

        shutil.copyfile(SMALL_DB, wrong)
        conn = sqlite3.connect(wrong)
        conn.execute("UPDATE tensors SET data = ? WHERE name = ?",
                     (np.zeros_like(arr).tobytes(), name))
        conn.commit()
        conn.close()

        try:
            apply_delta(wrong, delta, str(Path(tmp) / "bad.db"), verify=True, verbose=False)
            fail("mismatched base was accepted")
        except ValueError as e:
            check("does not match" in str(e), "mismatched base rejected")


# ── 6. Quantize model (needs SmolLM DB) ─────────────────────────────────

def test_quantize_model():
    section("Quantize model")
    if not SMALL_DB.exists():
        skip(f"{SMALL_DB.name} not present")
        return

    from reminis.quantize import quantize_model

    with tempfile.TemporaryDirectory() as tmp:
        for bits in (8, 4):
            out = str(Path(tmp) / f"q{bits}.db")
            stats = quantize_model(str(SMALL_DB), out, bits=bits, verbose=False)
            check(stats["quantized"] > 0, f"Q{bits}: tensors were quantized")
            check(stats["copied"] > 0, f"Q{bits}: norms copied through")

            a = sqlite3.connect(str(SMALL_DB))
            b = sqlite3.connect(out)
            n_a = a.execute("SELECT COUNT(*) FROM tensors").fetchone()[0]
            n_b = b.execute("SELECT COUNT(*) FROM tensors").fetchone()[0]
            check(n_a == n_b, f"Q{bits}: all tensors present")
            a.close()
            b.close()


# ── 7. Merge (needs SmolLM DB) ───────────────────────────────────────────

def test_merge():
    section("Merge")
    if not SMALL_DB.exists():
        skip(f"{SMALL_DB.name} not present")
        return

    from reminis.merge import merge_models

    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "merged.db")
        merge_models([str(SMALL_DB), str(SMALL_DB)], out,
                     method="linear", verbose=False)

        a = sqlite3.connect(str(SMALL_DB))
        b = sqlite3.connect(out)
        n_a = a.execute("SELECT COUNT(*) FROM tensors").fetchone()[0]
        n_b = b.execute("SELECT COUNT(*) FROM tensors").fetchone()[0]
        check(n_a == n_b, "linear merge preserves tensor count")
        a.close()
        b.close()


# ── 8. Registry (needs SmolLM DB) ────────────────────────────────────────

def test_registry():
    section("Registry")
    if not SMALL_DB.exists():
        skip(f"{SMALL_DB.name} not present")
        return

    from reminis.registry import Registry

    with tempfile.TemporaryDirectory() as tmp:
        reg_path = str(Path(tmp) / "reg.db")
        reg = Registry(reg_path, create=True)
        reg.add_base(str(SMALL_DB), "base", verbose=False)
        models = reg.list_models()
        check(len(models) == 1, "registry holds the added model")
        check(models[0]["name"] == "base", "model name is correct")

        out = str(Path(tmp) / "exported.db")
        reg.materialize("base", out, verbose=False)
        a = sqlite3.connect(str(SMALL_DB))
        b = sqlite3.connect(out)
        n_a = a.execute("SELECT COUNT(*) FROM tensors").fetchone()[0]
        n_b = b.execute("SELECT COUNT(*) FROM tensors").fetchone()[0]
        check(n_a == n_b, "materialized export has all tensors")
        a.close()
        b.close()

        reg.remove("base")
        check(len(reg.list_models()) == 0, "model removed")
        reg.close()


# ── 9. Lossy (low-rank) delta (needs SmolLM DB) ─────────────────────────

def test_lowrank():
    section("Lossy (low-rank) delta")
    if not SMALL_DB.exists():
        skip(f"{SMALL_DB.name} not present")
        return

    from reminis.diff import apply_delta, diff_models

    with tempfile.TemporaryDirectory() as tmp:
        target = str(Path(tmp) / "target.db")
        delta = str(Path(tmp) / "delta.db")
        result = str(Path(tmp) / "result.db")

        shutil.copyfile(SMALL_DB, target)

        conn = sqlite3.connect(target)
        row = conn.execute(
            "SELECT name, shape, dtype, data FROM tensors "
            "WHERE dtype IN ('F16','F32') AND shape LIKE '[%,%' LIMIT 1"
        ).fetchone()
        if row is None:
            skip("no 2D float tensor for low-rank test")
            conn.close()
            return

        name, shape_json, dtype, blob = row
        np_dtype = np.float32 if dtype == "F32" else np.float16
        shape = json.loads(shape_json)
        arr = np.frombuffer(blob, dtype=np_dtype).astype(np.float32)
        m, n = shape[1], shape[0]

        rng = np.random.default_rng(7)
        u = rng.normal(0, 0.01, (m, 2)).astype(np.float32)
        v = rng.normal(0, 0.01, (2, n)).astype(np.float32)
        perturbed = (arr.reshape(m, n) + u @ v).astype(np_dtype)
        conn.execute("UPDATE tensors SET data = ? WHERE name = ?",
                     (perturbed.tobytes(), name))
        conn.commit()
        conn.close()

        summary = diff_models(str(SMALL_DB), target, delta,
                              lossy_tolerance=0.01, verbose=False)
        check(summary["lossy"], "low-rank encoding was used")

        apply_delta(str(SMALL_DB), delta, result, verify=True, verbose=False)
        ok("lossy apply + verify succeeded")


# ── 10. Inference (needs SmolLM DB) ──────────────────────────────────────

def test_inference():
    section("Inference")
    if not SMALL_DB.exists():
        skip(f"{SMALL_DB.name} not present")
        return

    from reminis.infer import generate

    result = generate(str(SMALL_DB), "The capital of France is",
                      max_tokens=8, temperature=0.0, verbose=False)
    text = result.get("completion", "")
    check(len(text.strip()) > 0, "model generates text")
    check(all(ord(c) < 0x3000 for c in text),
          "output is not mojibake (block layout OK)")


# ── 11. Threaded weights hash consistency ────────────────────────────────

def test_encode_candidates_parallel():
    """Compressing the candidates on threads must not change a single byte.

    zstd's compressor objects are not thread-safe, and a shared one does not
    fail when used from several threads -- it produces corrupt or differing
    output. Both the encoding chosen and the payload are compared against
    the single-threaded result here, over every shape of delta.
    """
    section("Parallel candidate encoding")
    from concurrent.futures import ThreadPoolExecutor

    import reminis.diff as diff

    rng = np.random.default_rng(5)
    n = 40000
    base = rng.integers(0, 256, n, dtype=np.uint8).tobytes()

    mostly_same = bytearray(base)
    for i in range(0, n, 3):
        mostly_same[i] ^= 0x11

    cases = []
    for dtype in ("F16", "BF16", "F32", "Q4_K"):
        cases.append((f"{dtype} mostly unchanged", base, bytes(mostly_same), dtype))
        cases.append((f"{dtype} fully random", base,
                      rng.integers(0, 256, n, dtype=np.uint8).tobytes(), dtype))
        cases.append((f"{dtype} identical", base, base, dtype))
        cases.append((f"{dtype} size mismatch", base, base[: n // 2], dtype))

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        for label, a, b, dtype in cases:
            enc_serial, payload_serial = diff._encode_delta(a, b, dtype, None)
            enc_pooled, payload_pooled = diff._encode_delta(a, b, dtype, pool)
            check(enc_serial == enc_pooled and payload_serial == payload_pooled,
                  f"{label}: pooled encode is byte-identical ({enc_serial})")
    finally:
        pool.shutdown()


def test_diff_tensor_set_changes():
    """A pack must record the same hashes however the tensor sets line up.

    diff accumulates both weight hashes as it walks the shared tensors,
    which is only equivalent to hashing each model separately when neither
    side has a tensor the other lacks. Both arrangements are checked here,
    against _weights_hash itself.

    The added-tensor case also covers a bug that shipped through 0.28.0:
    the insert for a target-only tensor named nine columns and bound eight,
    so any diff that added a tensor raised OperationalError.
    """
    section("Diff across changing tensor sets")
    if not SMALL_DB.exists():
        skip(f"{SMALL_DB.name} not present")
        return

    from reminis.diff import _weights_hash, apply_delta, diff_models

    def truth(db):
        c = sqlite3.connect(db)
        h = _weights_hash(c)
        c.close()
        return h

    def pack_meta(p):
        c = sqlite3.connect(p)
        m = dict(c.execute("SELECT key, value FROM delta_meta"))
        c.close()
        return m

    for label, mutate in (
        ("same tensor set", None),
        ("target adds and drops a tensor", "reshape"),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "target.db")
            shutil.copyfile(SMALL_DB, target)

            conn = sqlite3.connect(target)
            name = conn.execute(
                "SELECT name FROM tensors WHERE dtype = 'F16' LIMIT 1"
            ).fetchone()[0]
            blob = conn.execute(
                "SELECT data FROM tensors WHERE name = ?", (name,)
            ).fetchone()[0]
            arr = np.frombuffer(blob, dtype=np.float16).copy()
            arr[:64] += np.float16(0.01)
            conn.execute("UPDATE tensors SET data = ? WHERE name = ?",
                         (arr.tobytes(), name))
            if mutate == "reshape":
                victim = conn.execute(
                    "SELECT name FROM tensors WHERE dtype = 'F16' AND name != ? "
                    "LIMIT 1", (name,)
                ).fetchone()[0]
                conn.execute("DELETE FROM tensors WHERE name = ?", (victim,))
                added = np.ones(64, dtype=np.float16)
                conn.execute(
                    "INSERT INTO tensors (name, shape, dtype, dtype_id, "
                    "n_elements, n_bytes, data) VALUES (?,?,?,?,?,?,?)",
                    ("brand.new.weight", "[64]", "F16", 1, 64,
                     added.nbytes, added.tobytes()),
                )
            conn.commit()
            conn.close()

            pack = str(Path(tmp) / "pack.db")
            diff_models(str(SMALL_DB), target, pack, verbose=False)
            meta = pack_meta(pack)
            check(meta["base_weights_hash"] == truth(str(SMALL_DB)),
                  f"{label}: base hash matches _weights_hash")
            check(meta["target_weights_hash"] == truth(target),
                  f"{label}: target hash matches _weights_hash")

            result = str(Path(tmp) / "result.db")
            apply_delta(str(SMALL_DB), pack, result, verify=True, verbose=False)
            check(truth(result) == truth(target),
                  f"{label}: pack rebuilds the target exactly")


def test_quantize_chunking():
    """Splitting a tensor across threads must change nothing about the bytes."""
    section("Chunked quantization")
    import reminis.quantize as quantize

    rng = np.random.default_rng(11)
    block = quantize.BLOCK
    n = quantize.PARALLEL_MIN_BLOCKS + 777  # enough to span several chunks

    cases = {
        "normal": rng.normal(0, 0.02, block * n).astype(np.float32),
        "all negative": -np.abs(rng.normal(0, 1, block * n)).astype(np.float32),
        "with zero blocks": np.concatenate([
            np.zeros(block * 64, dtype=np.float32),
            rng.normal(0, 1, block * (n - 64)).astype(np.float32),
        ]),
    }

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        for label, arr in cases.items():
            for name, fn in (("Q8_0", quantize.quantize_q8_0),
                             ("Q4_0", quantize.quantize_q4_0)):
                chunked = quantize._quantize_chunked(arr, fn, pool)
                check(chunked == fn(arr),
                      f"{name}, {label}: chunked output matches one pass")
    finally:
        pool.shutdown()

    # And end to end, through the real model path.
    if SMALL_DB.exists():
        with tempfile.TemporaryDirectory() as tmp:
            parallel = str(Path(tmp) / "par.db")
            quantize.quantize_model(str(SMALL_DB), parallel, bits=4, verbose=False)

            saved = quantize.PARALLEL_MIN_BLOCKS
            quantize.PARALLEL_MIN_BLOCKS = 1 << 62  # force the serial path
            try:
                serial = str(Path(tmp) / "ser.db")
                quantize.quantize_model(str(SMALL_DB), serial, bits=4, verbose=False)
            finally:
                quantize.PARALLEL_MIN_BLOCKS = saved

            import hashlib

            def digest(db):
                c = sqlite3.connect(db)
                out = {
                    n: hashlib.sha256(d).hexdigest()
                    for n, d in c.execute("SELECT name, data FROM tensors")
                }
                c.close()
                return out

            check(digest(parallel) == digest(serial),
                  "quantize_model: threaded and serial write the same tensors")


def test_gguf_fast_parser():
    """The fast header parser must agree with the reference reader exactly."""
    section("Fast GGUF header parser")
    if not SMALL_GGUF.exists():
        skip(f"{SMALL_GGUF.name} not present")
        return

    import mmap

    from gguf.constants import GGUFValueType
    from gguf.gguf_reader import GGUFReader

    from reminis.converter import (
        _extract_field_value, _meta_type_name, _parse_gguf_header,
    )

    reader = GGUFReader(str(SMALL_GGUF), mode="r")
    ref_meta = {}
    for key, field in reader.fields.items():
        main = field.types[0] if field.types else GGUFValueType.STRING
        elem = field.types[1] if len(field.types) > 1 else None
        ref_meta[key] = (str(_extract_field_value(field)),
                         _meta_type_name(main, elem))
    ref_tensors = {
        t.name: ([int(x) for x in t.shape], t.tensor_type.name,
                 int(t.tensor_type), int(t.n_elements), int(t.n_bytes),
                 int(t.data_offset))
        for t in reader.tensors
    }

    with open(SMALL_GGUF, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            meta, tensors, data_offset = _parse_gguf_header(buf)

    fast_meta = {k: (v, t) for k, v, t in meta}
    fast_tensors = {
        t["name"]: (t["shape"], t["dtype"], t["dtype_id"], t["n_elements"],
                    t["n_bytes"], data_offset + t["rel_offset"])
        for t in tensors
    }

    check(ref_meta == fast_meta,
          f"metadata matches the reference reader ({len(ref_meta)} fields)")
    check(ref_tensors == fast_tensors,
          f"tensor records match the reference reader ({len(ref_tensors)} tensors)")


def test_convert_fallback():
    """A file the fast parser declines still converts, and to the same bytes."""
    section("Conversion fallback")
    if not SMALL_GGUF.exists():
        skip(f"{SMALL_GGUF.name} not present")
        return

    import hashlib

    import reminis.converter as converter

    def snapshot(db):
        c = sqlite3.connect(db)
        meta = dict((k, (v, t)) for k, v, t
                    in c.execute("SELECT key, value, type FROM model_meta"))
        tensors = {
            n: (s, d, i, e, b, hashlib.sha256(x).hexdigest())
            for n, s, d, i, e, b, x in c.execute(
                "SELECT name, shape, dtype, dtype_id, n_elements, n_bytes, data "
                "FROM tensors")
        }
        c.close()
        return meta, tensors

    with tempfile.TemporaryDirectory() as tmp:
        fast_db = str(Path(tmp) / "fast.db")
        converter.gguf_to_sqlite(str(SMALL_GGUF), fast_db, verbose=False)

        original = converter._parse_gguf_header
        converter._parse_gguf_header = lambda buf: (_ for _ in ()).throw(
            ValueError("forced fallback"))
        try:
            slow_db = str(Path(tmp) / "slow.db")
            converter.gguf_to_sqlite(str(SMALL_GGUF), slow_db, verbose=False)
        finally:
            converter._parse_gguf_header = original

        fast_meta, fast_tensors = snapshot(fast_db)
        slow_meta, slow_tensors = snapshot(slow_db)
        check(fast_meta == slow_meta, "fallback writes the same metadata")
        check(fast_tensors == slow_tensors, "fallback writes the same tensors")


def test_every_subcommand_reaches_a_handler():
    """Each subcommand must set `func`, and each handler must be callable.

    The dispatch used to be `elif args.command == "..."` at the bottom of a
    495-line function, where a subcommand whose branch was missing parsed
    fine and then did nothing at all. Attaching handlers with `set_defaults`
    makes that a missing attribute instead -- but only if every subparser
    actually got one, which is what this checks.
    """
    section("CLI dispatch")
    import inspect

    from reminis.cli import build_parser

    parser = build_parser()
    subparsers = [a for a in parser._actions
                  if isinstance(a, argparse._SubParsersAction)]
    check(len(subparsers) == 1, "one top-level subparser group")

    commands = subparsers[0].choices
    check(len(commands) >= 17, f"{len(commands)} subcommands found")

    for name, sub in sorted(commands.items()):
        handler = sub.get_default("func")
        check(handler is not None, f"`{name}` reaches a handler")
        if handler is None:
            continue
        check(callable(handler), f"`{name}` handler is callable")
        # Every handler takes (args, parser): the parser so it can call
        # parser.error for combinations argparse cannot express.
        params = list(inspect.signature(handler).parameters)
        check(params == ["args", "parser"],
              f"`{name}` handler takes (args, parser)"
              + ("" if params == ["args", "parser"] else f" -- takes {params}"))

    # `registry` nests its own subcommands, which dispatch through
    # _run_registry rather than through set_defaults. Check they parse.
    reg = commands["registry"]
    reg_subs = [a for a in reg._actions
                if isinstance(a, argparse._SubParsersAction)]
    check(len(reg_subs) == 1, "registry has its own subparser group")
    check(set(reg_subs[0].choices) == {"add", "add-lora", "ls", "export", "rm"},
          f"registry subcommands: {sorted(reg_subs[0].choices)}")
    check(reg.get_default("registry_parser") is reg,
          "registry carries its own parser through for bare-command help")

    # A no-argument invocation must still print help and exit 1 rather than
    # raising AttributeError on the missing `func`.
    result = subprocess.run(
        [sys.executable, "-c", "from reminis.cli import main; main()"],
        capture_output=True, text=True,
    )
    check(result.returncode == 1, f"bare `reminis` exits 1, got {result.returncode}")
    check("usage:" in (result.stdout + result.stderr), "bare `reminis` prints usage")


def test_tensor_sql_matches_the_schema():
    """Every generated statement must agree with the table it writes to.

    A delta pack insert named nine columns and bound eight for several
    releases, failing only on the one path that added a tensor rather than
    changing one. The statements are generated from `TENSOR_COLUMNS` now, so
    the counts cannot drift -- but nothing yet checked that tuple against the
    schema itself, which is the half that would still go wrong silently.
    """
    section("Tensor SQL agrees with the schema")
    from reminis.converter import SCHEMA
    from reminis.db import (
        INSERT_OR_REPLACE_TENSOR,
        INSERT_REGISTRY_TENSOR,
        INSERT_TENSOR,
        TENSOR_COLUMNS,
        UPSERT_TENSOR,
    )
    from reminis.registry import REGISTRY_SCHEMA

    statements = {
        "INSERT_TENSOR": (INSERT_TENSOR, SCHEMA, len(TENSOR_COLUMNS)),
        "INSERT_OR_REPLACE_TENSOR": (INSERT_OR_REPLACE_TENSOR, SCHEMA, len(TENSOR_COLUMNS)),
        "UPSERT_TENSOR": (UPSERT_TENSOR, SCHEMA, len(TENSOR_COLUMNS)),
        "INSERT_REGISTRY_TENSOR": (INSERT_REGISTRY_TENSOR, REGISTRY_SCHEMA,
                                   len(TENSOR_COLUMNS) + 1),
    }

    for label, (sql, schema, expected) in statements.items():
        check(sql.count("?") == expected,
              f"{label} binds {expected} values")

        # SQLite is the authority on whether the statement fits the table, so
        # ask it rather than comparing strings: prepare against a real schema.
        with tempfile.TemporaryDirectory() as tmp:
            conn = sqlite3.connect(str(Path(tmp) / "s.db"))
            conn.executescript(schema)
            try:
                conn.execute(sql, (None,) * expected)
            except sqlite3.IntegrityError:
                # NOT NULL rejects the dummy row, which means the statement
                # itself parsed and bound cleanly. That is what is being
                # checked here.
                check(True, f"{label} matches its table")
            except sqlite3.Error as e:
                check(False, f"{label} matches its table -- {e}")
            else:
                check(False, f"{label}: NOT NULL should have rejected a null row")
            conn.close()

    # The column tuple must be the schema's tensor columns, in order. Anything
    # added to one and not the other shows up here rather than at runtime.
    with tempfile.TemporaryDirectory() as tmp:
        conn = sqlite3.connect(str(Path(tmp) / "s.db"))
        conn.executescript(SCHEMA)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(tensors)") if r[1] != "id"]
        conn.close()
    check(tuple(cols) == TENSOR_COLUMNS,
          f"TENSOR_COLUMNS is the schema's column list ({cols})")


def test_threaded_hash():
    section("Threaded weights hash")
    if not SMALL_DB.exists():
        skip(f"{SMALL_DB.name} not present")
        return

    from reminis.diff import _weights_hash, _weights_hash_threaded

    conn = sqlite3.connect(str(SMALL_DB))
    h_seq = _weights_hash(conn)
    conn.close()

    h_threaded = _weights_hash_threaded(str(SMALL_DB))
    check(h_seq == h_threaded, "threaded hash matches sequential hash")


# ── runner ────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_bitplane,
    test_quantize_blocks,
    test_viewer,
    test_gguf_roundtrip,
    test_diff_apply,
    test_quantize_model,
    test_merge,
    test_registry,
    test_lowrank,
    test_inference,
    test_encode_candidates_parallel,
    test_diff_tensor_set_changes,
    test_quantize_chunking,
    test_gguf_fast_parser,
    test_convert_fallback,
    test_every_subcommand_reaches_a_handler,
    test_tensor_sql_matches_the_schema,
    test_threaded_hash,
]


def main():
    import time
    t0 = time.perf_counter()

    print("reminis pre-release test suite")
    print(f"model: {SMALL_DB.name} ({'present' if SMALL_DB.exists() else 'MISSING'})")

    for test in ALL_TESTS:
        try:
            test()
        except Exception as exc:
            fail(f"{test.__name__} raised: {exc}")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 70}")
    print(f"  {passed} passed, {failed} failed, {skipped} skipped  ({elapsed:.1f}s)")
    print(f"{'=' * 70}")

    if failed:
        print("\nPRE-RELEASE CHECK FAILED")
        sys.exit(1)
    else:
        print("\nPRE-RELEASE CHECK PASSED")


if __name__ == "__main__":
    main()
