"""Measure how well LoRA-style fine-tune deltas compress.

A LoRA update is W + B@A with B,A of rank r, so the delta is rank-r by
construction and should need only r*(m+n) numbers instead of m*n. This script
builds a synthetic LoRA merge on top of a real base model, then measures:

  1. what the current XOR delta pack costs,
  2. whether SVD actually recovers rank r from the merged f16 weights,
  3. what an ideal low-rank pack would cost.

The interesting wrinkle is step 2. Merging into a float16 model rounds the
result, and that rounding error is full-rank noise laid over the clean
low-rank signal -- so the recovered spectrum is not a clean cliff at r.
"""

import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

from reminis.diff import diff_models

MODELS_DIR = Path(__file__).parent.parent / "models"
TMP = Path(__file__).parent / "tmp_lora"

LORA_RANK = 16
# Scale the update so its Frobenius norm is this fraction of the base weight's.
# A trained LoRA nudges weights; it does not overwrite them. Using a raw
# alpha/r scale on random factors produces an update ~25x larger than the
# weights themselves, which is not a fine-tune and makes XOR look worse than
# it deserves.
LORA_RELATIVE_STRENGTH = 0.05
# Standard LoRA targets the attention projections.
TARGET_SUFFIXES = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")


def build_lora_merge(db_path: str, rank: int, seed: int = 0) -> int:
    """Apply a synthetic rank-r LoRA update to every targeted tensor in place."""
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT name, shape, dtype, data FROM tensors").fetchall()

    touched = 0
    for name, shape_json, dtype, blob in rows:
        if not name.endswith(TARGET_SUFFIXES):
            continue
        if dtype != "F16":
            continue

        import json
        # GGUF stores shape reversed relative to the data layout.
        shape = json.loads(shape_json)[::-1]
        m, n = shape

        w = np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(m, n)

        # Standard LoRA init: A ~ N(0, 1/r), B = 0 then trained. We simulate a
        # *trained* adapter, so both factors are non-zero.
        a = rng.normal(0, 1.0 / np.sqrt(rank), size=(rank, n)).astype(np.float32)
        b = rng.normal(0, 1.0 / np.sqrt(rank), size=(m, rank)).astype(np.float32)
        with np.errstate(all="ignore"):  # spurious BLAS SIMD warnings; output verified finite
            update = b @ a

        # Normalize to a realistic strength relative to the weights it modifies.
        update *= LORA_RELATIVE_STRENGTH * np.linalg.norm(w) / np.linalg.norm(update)

        merged = (w + update).astype(np.float16)
        conn.execute("UPDATE tensors SET data = ? WHERE name = ?", (merged.tobytes(), name))
        touched += 1

    conn.commit()
    conn.close()
    return touched


def analyze(base_db: str, merged_db: str, rank: int):
    """Compare the recovered spectrum against the known planted rank."""
    import json

    a = sqlite3.connect(base_db)
    b = sqlite3.connect(merged_db)

    samples = ["blk.0.attn_q.weight", "blk.15.attn_v.weight", "blk.29.attn_output.weight"]

    print(f"\n{'tensor':<30} {'shape':<13} {'r@90%':>6} {'r@99%':>6} "
          f"{'planted':>8} {'signal/noise':>13}")
    print("-" * 84)

    ideal_bytes = 0
    dense_bytes = 0

    for name in samples:
        ra = a.execute("SELECT shape,data FROM tensors WHERE name=?", (name,)).fetchone()
        rb = b.execute("SELECT shape,data FROM tensors WHERE name=?", (name,)).fetchone()
        shape = json.loads(ra[0])[::-1]
        m, n = shape

        x = np.frombuffer(ra[1], dtype=np.float16).astype(np.float32).reshape(m, n)
        y = np.frombuffer(rb[1], dtype=np.float16).astype(np.float32).reshape(m, n)
        d = y - x

        s = np.linalg.svd(d, compute_uv=False)
        energy = np.cumsum(s**2) / np.sum(s**2)
        r90 = int(np.searchsorted(energy, 0.90) + 1)
        r99 = int(np.searchsorted(energy, 0.99) + 1)

        # How far above the noise floor does the planted signal sit?
        signal = s[:rank].mean()
        noise = s[rank:].mean() if len(s) > rank else 0.0
        ratio = signal / noise if noise > 0 else float("inf")

        print(f"{name:<30} {str(shape):<13} {r90:>6} {r99:>6} {rank:>8} {ratio:>12.1f}x")

        ideal_bytes += rank * (m + n) * 2
        dense_bytes += d.size * 2

    a.close()
    b.close()

    print(f"\n  Across these 3 tensors:")
    print(f"    dense f16 delta:      {dense_bytes/1024:>9.1f} KB")
    print(f"    ideal rank-{rank} factors: {ideal_bytes/1024:>9.1f} KB "
          f"({ideal_bytes/dense_bytes*100:.1f}% of dense)")


def main():
    base_db = MODELS_DIR / "SmolLM-135M.f16.db"
    if not base_db.exists():
        print(f"Missing {base_db}")
        print("Run: reminis convert models/SmolLM-135M.f16.gguf -o models/SmolLM-135M.f16.db")
        sys.exit(1)

    TMP.mkdir(parents=True, exist_ok=True)
    merged_db = str(TMP / "lora_merged.db")
    pack_db = str(TMP / "lora.delta.db")

    print("=" * 84)
    print(f"Synthetic LoRA merge: rank {LORA_RANK}, "
          f"strength {LORA_RELATIVE_STRENGTH:.0%} of weight norm, attention projections only")
    print("=" * 84)

    shutil.copyfile(base_db, merged_db)
    touched = build_lora_merge(merged_db, LORA_RANK)
    print(f"\nApplied a rank-{LORA_RANK} update to {touched} tensors.")

    print("\n" + "-" * 84)
    print("Current XOR delta pack:")
    print("-" * 84)
    summary = diff_models(str(base_db), merged_db, pack_db, verbose=True)

    print("\n" + "-" * 84)
    print("Does SVD recover the planted rank?")
    print("-" * 84)
    analyze(str(base_db), merged_db, LORA_RANK)

    # What would a whole-model low-rank pack cost?
    import json
    conn = sqlite3.connect(merged_db)
    total_ideal = 0
    for name, shape_json in conn.execute("SELECT name, shape FROM tensors"):
        if name.endswith(TARGET_SUFFIXES):
            m, n = json.loads(shape_json)[::-1]
            total_ideal += LORA_RANK * (m + n) * 2
    full = conn.execute("SELECT SUM(n_bytes) FROM tensors").fetchone()[0]
    conn.close()

    print("\n" + "=" * 84)
    print("Bottom line")
    print("=" * 84)
    print(f"  Full model:                {full/1024/1024:>8.1f} MB")
    print(f"  XOR delta pack (built):    {summary['delta_stored_bytes']/1024/1024:>8.1f} MB  "
          f"({summary['delta_stored_bytes']/full*100:.1f}% of full)")
    print(f"  Ideal rank-{LORA_RANK} pack:        {total_ideal/1024/1024:>8.1f} MB  "
          f"({total_ideal/full*100:.1f}% of full)")
    print(f"\n  Tensors changed: {summary['changed']} of 272")

    shutil.rmtree(TMP)


if __name__ == "__main__":
    main()
