"""Benchmark the slow paths in reminis.

Run:  python bench.py [--model path/to/model.db] [--only name,name,...]

Each benchmark prints a one-line result with the metric that matters,
so progress across the day shows up in a simple diff.
"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL = MODELS_DIR / "llama1b.db"
GGUF_MODEL = MODELS_DIR / "Llama-3.2-1B-Instruct-f16.gguf"

RESULTS = []


def record(name, elapsed, unit, detail=""):
    RESULTS.append({"name": name, "elapsed": elapsed, "unit": unit, "detail": detail})
    tag = f"  ({detail})" if detail else ""
    print(f"  {name:<40s} {elapsed:>8.3f} {unit}{tag}")


# ── 1. GGUF → SQLite conversion ──────────────────────────────────────────

def bench_gguf_to_sqlite(gguf_path):
    from reminis.converter import gguf_to_sqlite

    with tempfile.TemporaryDirectory(prefix="reminis-bench-") as tmp:
        out = os.path.join(tmp, "converted.db")
        t0 = time.perf_counter()
        gguf_to_sqlite(str(gguf_path), out, verbose=False)
        elapsed = time.perf_counter() - t0
        size_mb = Path(out).stat().st_size / 1e6
        record("gguf_to_sqlite", elapsed, "s", f"{size_mb:.0f} MB, {size_mb/elapsed:.0f} MB/s")


# ── 2. SQLite → GGUF export ──────────────────────────────────────────────

def bench_sqlite_to_gguf(db_path):
    from reminis.converter import sqlite_to_gguf

    with tempfile.TemporaryDirectory(prefix="reminis-bench-") as tmp:
        out = os.path.join(tmp, "exported.gguf")
        t0 = time.perf_counter()
        sqlite_to_gguf(str(db_path), out, verbose=False)
        elapsed = time.perf_counter() - t0
        size_mb = Path(out).stat().st_size / 1e6
        record("sqlite_to_gguf", elapsed, "s", f"{size_mb:.0f} MB, {size_mb/elapsed:.0f} MB/s")


# ── 3. Weights hash (SHA-256 over all tensor blobs) ──────────────────────

def bench_weights_hash(db_path):
    conn = sqlite3.connect(str(db_path))
    t0 = time.perf_counter()
    h = hashlib.sha256()
    for (blob,) in conn.execute("SELECT data FROM tensors ORDER BY name"):
        h.update(blob)
    elapsed = time.perf_counter() - t0
    conn.close()
    total_mb = Path(db_path).stat().st_size / 1e6
    record("weights_hash", elapsed, "s", f"{total_mb:.0f} MB, {total_mb/elapsed:.0f} MB/s")


# ── 4. Quantization (Q8_0 and Q4_0) ─────────────────────────────────────

def bench_quantize(db_path):
    from reminis.quantize import quantize_model

    for bits in (8, 4):
        with tempfile.TemporaryDirectory(prefix="reminis-bench-") as tmp:
            out = os.path.join(tmp, f"q{bits}.db")
            t0 = time.perf_counter()
            stats = quantize_model(str(db_path), out, bits=bits, verbose=False)
            elapsed = time.perf_counter() - t0
            raw_mb = stats["raw_bytes"] / 1e6
            new_mb = stats["new_bytes"] / 1e6
            record(f"quantize_Q{bits}_0", elapsed, "s",
                   f"{raw_mb:.0f} → {new_mb:.0f} MB, {raw_mb/elapsed:.0f} MB/s")


# ── 5. Diff (lossless delta pack) ────────────────────────────────────────

def bench_diff(db_path):
    from reminis.diff import diff_models

    with tempfile.TemporaryDirectory(prefix="reminis-bench-") as tmp:
        out = os.path.join(tmp, "delta.db")
        t0 = time.perf_counter()
        summary = diff_models(str(db_path), str(db_path), out, verbose=False)
        elapsed = time.perf_counter() - t0
        record("diff_identical", elapsed, "s",
               f"{summary['shared']} tensors, all identical")


# ── 6. Apply delta ───────────────────────────────────────────────────────

def bench_apply_delta(db_path):
    """Create a real delta pack from a quantized variant, then apply it."""
    from reminis.diff import diff_models, apply_delta
    from reminis.quantize import quantize_model

    with tempfile.TemporaryDirectory(prefix="reminis-bench-") as tmp:
        q8_path = os.path.join(tmp, "q8.db")
        quantize_model(str(db_path), q8_path, bits=8, verbose=False)

        delta_path = os.path.join(tmp, "delta.db")
        diff_models(str(db_path), q8_path, delta_path, verbose=False)
        delta_size = Path(delta_path).stat().st_size / 1e6

        out_path = os.path.join(tmp, "applied.db")
        t0 = time.perf_counter()
        apply_delta(str(db_path), delta_path, out_path, verify=True, verbose=False)
        elapsed = time.perf_counter() - t0
        record("apply_delta_verified", elapsed, "s",
               f"delta {delta_size:.0f} MB")


# ── 7. Inference: prefill ────────────────────────────────────────────────

def bench_prefill(db_path):
    from reminis.infer import Model, KVCache
    from reminis.backend import select as select_backend

    backend = select_backend("inference")
    model = Model(str(db_path), backend=backend)
    tokens = model.tokenizer.encode(
        "The capital of France is Paris. The capital of Japan is Tokyo. "
        "Machine learning models are trained on large datasets."
    )
    try:
        cache = KVCache(model.cfg.n_layers, capacity=len(tokens) + 16,
                        backend=model.backend)
        t0 = time.perf_counter()
        model.forward(tokens, cache, offset=0)
        elapsed = time.perf_counter() - t0
        record("prefill", elapsed, "s",
               f"{len(tokens)} tokens, {len(tokens)/elapsed:.1f} tok/s")
    finally:
        model.close()


# ── 8. Inference: decode (token generation) ──────────────────────────────

def bench_decode(db_path, n_tokens=16):
    from reminis.infer import Model, KVCache, _sample
    from reminis.backend import select as select_backend

    backend = select_backend("inference")
    model = Model(str(db_path), backend=backend)
    prompt = "The meaning of life is"
    tokens = model.tokenizer.encode(prompt)
    rng = np.random.default_rng(42)

    try:
        cache = KVCache(model.cfg.n_layers, capacity=len(tokens) + n_tokens + 1,
                        backend=model.backend)
        logits = model.forward(tokens, cache, offset=0)

        t0 = time.perf_counter()
        for _ in range(n_tokens):
            token_id = _sample(logits, 0.8, 0.95, rng)
            logits = model.forward([token_id], cache, offset=cache.length)
        elapsed = time.perf_counter() - t0
        record("decode", elapsed, "s",
               f"{n_tokens} tokens, {n_tokens/elapsed:.2f} tok/s")
    finally:
        model.close()


# ── 9. Inference: stream mode (re-read every weight) ─────────────────────

def bench_stream_decode(db_path, n_tokens=4):
    from reminis.infer import Model, KVCache, _sample
    from reminis.backend import select as select_backend

    backend = select_backend("inference")
    model = Model(str(db_path), stream=True, backend=backend)
    tokens = model.tokenizer.encode("Hello")
    rng = np.random.default_rng(42)

    try:
        cache = KVCache(model.cfg.n_layers, capacity=len(tokens) + n_tokens + 1,
                        backend=model.backend)
        logits = model.forward(tokens, cache, offset=0)

        t0 = time.perf_counter()
        for _ in range(n_tokens):
            token_id = _sample(logits, 0.8, 0.95, rng)
            logits = model.forward([token_id], cache, offset=cache.length)
        elapsed = time.perf_counter() - t0
        record("stream_decode", elapsed, "s",
               f"{n_tokens} tokens, {n_tokens/elapsed:.2f} tok/s")
    finally:
        model.close()


# ── 10. Raw SQLite blob read throughput ──────────────────────────────────

def bench_sqlite_read(db_path):
    conn = sqlite3.connect(str(db_path))
    total_bytes = 0
    t0 = time.perf_counter()
    for (blob,) in conn.execute("SELECT data FROM tensors"):
        total_bytes += len(blob)
    elapsed = time.perf_counter() - t0
    conn.close()
    mb = total_bytes / 1e6
    record("sqlite_blob_read", elapsed, "s", f"{mb:.0f} MB, {mb/elapsed:.0f} MB/s")


# ── 11. Tensor dequantization throughput ─────────────────────────────────

def bench_dequantize(db_path):
    from reminis.dtypes import to_float32_any, is_quantized_dtype

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT dtype, data FROM tensors WHERE dtype NOT IN ('F32', 'F16', 'BF16') LIMIT 20"
    ).fetchall()
    conn.close()

    if not rows:
        record("dequantize", 0.0, "s", "no quantized tensors to test")
        return

    total_bytes = sum(len(blob) for _, blob in rows)
    t0 = time.perf_counter()
    for dtype, blob in rows:
        try:
            to_float32_any(blob, dtype)
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    mb = total_bytes / 1e6
    record("dequantize", elapsed, "s",
           f"{len(rows)} tensors, {mb:.0f} MB, {mb/elapsed:.0f} MB/s")


# ── runner ────────────────────────────────────────────────────────────────

ALL_BENCHMARKS = {
    "sqlite_read":       lambda db, gguf: bench_sqlite_read(db),
    "weights_hash":      lambda db, gguf: bench_weights_hash(db),
    "gguf_to_sqlite":    lambda db, gguf: bench_gguf_to_sqlite(gguf) if gguf else None,
    "sqlite_to_gguf":    lambda db, gguf: bench_sqlite_to_gguf(db),
    "quantize":          lambda db, gguf: bench_quantize(db),
    "dequantize":        lambda db, gguf: bench_dequantize(db),
    "diff":              lambda db, gguf: bench_diff(db),
    "apply_delta":       lambda db, gguf: bench_apply_delta(db),
    "prefill":           lambda db, gguf: bench_prefill(db),
    "decode":            lambda db, gguf: bench_decode(db),
    "stream_decode":     lambda db, gguf: bench_stream_decode(db),
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark reminis bottlenecks")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to the model database")
    parser.add_argument("--gguf", type=Path, default=GGUF_MODEL,
                        help="Path to GGUF file for conversion benchmark")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of benchmarks to run")
    parser.add_argument("--json", type=Path, default=None,
                        help="Write results to this JSON file")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)

    gguf = args.gguf if args.gguf.exists() else None
    if gguf is None:
        print(f"Note: GGUF not found at {args.gguf}, skipping gguf_to_sqlite\n")

    selected = ALL_BENCHMARKS
    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        selected = {k: v for k, v in ALL_BENCHMARKS.items() if k in names}
        missing = set(names) - set(selected)
        if missing:
            print(f"Unknown benchmarks: {', '.join(missing)}")
            print(f"Available: {', '.join(ALL_BENCHMARKS)}")
            sys.exit(1)

    size_mb = args.model.stat().st_size / 1e6
    print(f"reminis benchmark")
    print(f"  model: {args.model.name} ({size_mb:.0f} MB)")
    print(f"  benchmarks: {', '.join(selected)}")
    print()

    for name, fn in selected.items():
        try:
            fn(args.model, gguf)
        except Exception as exc:
            record(name, 0.0, "s", f"FAILED: {exc}")

    print()
    print("=" * 72)
    print(f"{'benchmark':<40s} {'time':>8} {'detail'}")
    print("-" * 72)
    for r in RESULTS:
        tag = f"  {r['detail']}" if r["detail"] else ""
        print(f"  {r['name']:<40s} {r['elapsed']:>7.3f}{r['unit']}{tag}")
    print("=" * 72)

    if args.json:
        args.json.write_text(json.dumps({
            "model": str(args.model),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": RESULTS,
        }, indent=2))
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
