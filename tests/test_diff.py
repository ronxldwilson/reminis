"""Verify that diff produces delta packs which apply back to the exact target."""

import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

from reminis.diff import _weights_hash, apply_delta, diff_models

MODELS_DIR = Path(__file__).parent.parent / "models"
TMP_DIR = Path(__file__).parent / "tmp_diff"


def perturb(db_path: str, tensor_names: list[str], scale: float = 0.01, seed: int = 0):
    """Add small random noise to specific tensors, simulating a fine-tune."""
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(db_path)
    for name in tensor_names:
        row = conn.execute("SELECT dtype, data FROM tensors WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise ValueError(f"tensor not found: {name}")
        dtype, blob = row
        np_dtype = np.float32 if dtype == "F32" else np.float16
        arr = np.frombuffer(blob, dtype=np_dtype).astype(np.float32)
        noise = rng.normal(0, scale, size=arr.shape).astype(np.float32)
        new = (arr + noise).astype(np_dtype)
        conn.execute("UPDATE tensors SET data = ? WHERE name = ?", (new.tobytes(), name))
    conn.commit()
    conn.close()


def main():
    base_db = MODELS_DIR / "SmolLM-135M.f16.db"
    if not base_db.exists():
        print(f"Missing {base_db}. Run: reminis convert models/SmolLM-135M.f16.gguf")
        sys.exit(1)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    target_db = str(TMP_DIR / "target.db")
    delta_db = str(TMP_DIR / "delta.db")
    result_db = str(TMP_DIR / "result.db")

    print("=" * 70)
    print("Test: synthetic fine-tune (perturb 5 tensors) -> diff -> apply")
    print("=" * 70)

    shutil.copyfile(base_db, target_db)
    touched = [
        "blk.0.attn_q.weight",
        "blk.5.ffn_gate.weight",
        "blk.15.attn_output.weight",
        "blk.29.ffn_down.weight",
        "output_norm.weight",
    ]
    perturb(target_db, touched)
    print(f"\nPerturbed {len(touched)} tensors in the target.\n")

    summary = diff_models(str(base_db), target_db, delta_db, verbose=True)

    assert summary["changed"] == len(touched), (
        f"expected {len(touched)} changed tensors, got {summary['changed']}"
    )
    changed_names = {t["name"] for t in summary["changed_tensors"]}
    assert changed_names == set(touched), f"wrong tensors flagged: {changed_names}"
    print("\n  Diff detected exactly the perturbed tensors.")

    print("\n" + "-" * 70)
    apply_delta(str(base_db), delta_db, result_db, verify=True, verbose=True)

    conn_t = sqlite3.connect(target_db)
    conn_r = sqlite3.connect(result_db)
    hash_t = _weights_hash(conn_t)
    hash_r = _weights_hash(conn_r)
    conn_t.close()
    conn_r.close()

    assert hash_t == hash_r, f"result != target\n  target: {hash_t}\n  result: {hash_r}"
    print("\n  Result weights hash matches target exactly.")

    print("\n" + "=" * 70)
    print("Test: applying a pack to the wrong base is rejected")
    print("=" * 70)

    wrong_base = str(TMP_DIR / "wrong_base.db")
    shutil.copyfile(base_db, wrong_base)
    perturb(wrong_base, ["blk.1.attn_k.weight"], seed=99)

    try:
        apply_delta(wrong_base, delta_db, str(TMP_DIR / "bad.db"), verify=True, verbose=False)
        print("\n  FAIL: mismatched base was accepted")
        sys.exit(1)
    except ValueError as e:
        assert "does not match" in str(e)
        print("\n  Correctly rejected a mismatched base model.")

    shutil.rmtree(TMP_DIR)
    print("\n" + "=" * 70)
    print("ALL DIFF TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
