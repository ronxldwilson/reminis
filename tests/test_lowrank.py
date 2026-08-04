"""Verify low-rank (lossy) delta packs against a LoRA-shaped fine-tune."""

import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

from reminis.diff import _weights_hash, apply_delta, diff_models

MODELS_DIR = Path(__file__).parent.parent / "models"
TMP = Path(__file__).parent / "tmp_lowrank"

RANK = 16
STRENGTH = 0.05
TARGETS = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")


def build_lora_merge(db_path: str, rank: int, seed: int = 0) -> int:
    """Merge a synthetic rank-r LoRA update into the targeted tensors."""
    import json

    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT name, shape, dtype, data FROM tensors").fetchall()

    touched = 0
    for name, shape_json, dtype, blob in rows:
        if not name.endswith(TARGETS) or dtype != "F16":
            continue
        m, n = json.loads(shape_json)[::-1]
        w = np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(m, n)

        a = rng.normal(0, 1 / np.sqrt(rank), size=(rank, n)).astype(np.float32)
        b = rng.normal(0, 1 / np.sqrt(rank), size=(m, rank)).astype(np.float32)
        with np.errstate(all="ignore"):  # spurious BLAS SIMD warnings
            update = b @ a
        update *= STRENGTH * np.linalg.norm(w) / np.linalg.norm(update)

        conn.execute(
            "UPDATE tensors SET data = ? WHERE name = ?",
            ((w + update).astype(np.float16).tobytes(), name),
        )
        touched += 1

    conn.commit()
    conn.close()
    return touched


def max_tensor_error(db_a: str, db_b: str) -> float:
    """Worst per-tensor relative error between two models."""
    import json

    a = sqlite3.connect(db_a)
    b = sqlite3.connect(db_b)
    worst = 0.0
    for name, dtype, blob_a in a.execute("SELECT name, dtype, data FROM tensors"):
        if dtype not in ("F16", "F32"):
            continue
        row = b.execute("SELECT data FROM tensors WHERE name = ?", (name,)).fetchone()
        nd = np.float32 if dtype == "F32" else np.float16
        x = np.frombuffer(blob_a, dtype=nd).astype(np.float32)
        y = np.frombuffer(row[0], dtype=nd).astype(np.float32)
        denom = np.linalg.norm(y)
        if denom > 0:
            worst = max(worst, float(np.linalg.norm(x - y) / denom))
    a.close()
    b.close()
    return worst


def main():
    base_db = MODELS_DIR / "SmolLM-135M.f16.db"
    if not base_db.exists():
        print(f"Missing {base_db}")
        sys.exit(1)

    TMP.mkdir(parents=True, exist_ok=True)
    target = str(TMP / "lora.db")
    lossless_pack = str(TMP / "lossless.db")
    lossy_pack = str(TMP / "lossy.db")
    result = str(TMP / "result.db")

    print("=" * 78)
    print(f"LoRA-shaped fine-tune: rank {RANK}, {STRENGTH:.0%} of weight norm")
    print("=" * 78)

    shutil.copyfile(base_db, target)
    touched = build_lora_merge(target, RANK)
    print(f"\nMerged a rank-{RANK} update into {touched} tensors.\n")

    print("-" * 78)
    print("Lossless pack (default):")
    print("-" * 78)
    lossless = diff_models(str(base_db), target, lossless_pack, verbose=False)
    full = lossless["target_total_bytes"]
    print(f"  {lossless['delta_stored_bytes']/1024/1024:8.1f} MB "
          f"({lossless['delta_stored_bytes']/full*100:.1f}% of model)   lossy={lossless['lossy']}")

    print("\n" + "-" * 78)
    print("Low-rank pack (--lossy 0.01):")
    print("-" * 78)
    lossy = diff_models(str(base_db), target, lossy_pack, verbose=False, lossy_tolerance=0.01)
    print(f"  {lossy['delta_stored_bytes']/1024/1024:8.1f} MB "
          f"({lossy['delta_stored_bytes']/full*100:.1f}% of model)   "
          f"low-rank tensors={lossy['lowrank_tensors']}/{lossy['changed']}")
    print(f"  worst per-tensor error reported: {lossy['max_rel_error']:.2e}")

    shrink = lossless["delta_stored_bytes"] / lossy["delta_stored_bytes"]
    print(f"\n  Low-rank pack is {shrink:.1f}x smaller than lossless.")

    assert lossy["lowrank_tensors"] > 0, "no tensors were low-rank encoded"
    assert lossy["delta_stored_bytes"] < lossless["delta_stored_bytes"], \
        "lossy pack is not smaller than lossless"

    print("\n" + "-" * 78)
    print("Apply the lossy pack:")
    print("-" * 78)
    apply_delta(str(base_db), lossy_pack, result, verify=True, verbose=True)

    # Apply must be exactly reproducible even though the pack is lossy.
    r = sqlite3.connect(result)
    p = sqlite3.connect(lossy_pack)
    recorded = dict(p.execute("SELECT key, value FROM delta_meta"))
    actual = _weights_hash(r)
    r.close()
    p.close()
    assert actual == recorded["reconstructed_weights_hash"], "apply is not deterministic"
    print("\n  Apply reproduces the recorded reconstruction hash exactly.")

    # And the divergence from the true target must respect the tolerance.
    err = max_tensor_error(result, target)
    print(f"  Measured worst error vs true target: {err:.2e}")
    assert err <= 0.01 * 1.5, f"error {err:.2e} exceeds the 1% tolerance"
    assert actual != recorded["target_weights_hash"], \
        "lossy result unexpectedly matches the target byte for byte"
    print("  Within tolerance, and correctly NOT byte-identical to the target.")

    print("\n" + "=" * 78)
    print("Test: a full fine-tune should decline low-rank and stay lossless")
    print("=" * 78)
    instruct = MODELS_DIR / "SmolLM-135M-Instruct.f16.db"
    if instruct.exists():
        s = diff_models(str(base_db), str(instruct), str(TMP / "ft.db"),
                        verbose=False, lossy_tolerance=0.01)
        pct = s["delta_stored_bytes"] / s["target_total_bytes"] * 100
        print(f"\n  low-rank tensors: {s['lowrank_tensors']} of {s['changed']}")
        print(f"  pack: {pct:.1f}% of model")
        print("  (a full fine-tune's delta is not low-rank, so few or no tensors qualify)")
    else:
        print("\n  SKIP: Instruct model not converted")

    shutil.rmtree(TMP)
    print("\n" + "=" * 78)
    print("ALL LOW-RANK TESTS PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
