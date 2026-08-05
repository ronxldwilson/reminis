"""End-to-end LoRA check on a real model, against peft's own merge.

The unit test uses a toy Llama built in-process. This runs the same path on a
real downloaded checkpoint, which is where the name-matching actually gets
tested: peft's module paths have to line up with the tensor names reminis
imported from the real safetensors file, and a toy model cannot prove that.

The base here is BF16, so unlike the float32 toy case the merge genuinely
rounds -- this measures how far that rounding puts us from peft.

Not part of the test suite: it needs a downloaded model and the training
stack. Run it directly.
"""

import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reminis.diff import apply_delta
from reminis.dtypes import to_float32
from reminis.lora import lora_to_delta_pack
from reminis.safetensors_io import safetensors_to_sqlite

MODEL_DIR = Path(__file__).parent.parent / "models" / "SmolLM2-135M-st"
TMP = Path(__file__).parent / "tmp_lora_real"

RANK = 16
ALPHA = 32

try:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
except ImportError:
    print("SKIP: needs torch, transformers, and peft")
    sys.exit(0)

if not MODEL_DIR.exists():
    print(f"SKIP: {MODEL_DIR} not present")
    sys.exit(0)


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Real-model LoRA: {MODEL_DIR.name}, rank {RANK}, alpha {ALPHA}")
    print("=" * 78)

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    print(f"\n  loaded {sum(p.numel() for p in model.parameters()):,} parameters "
          f"({next(model.parameters()).dtype})")

    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )

    # peft zeroes lora_B, so an untrained adapter merges to a no-op. Randomise
    # both factors or every comparison below would pass trivially.
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.copy_(torch.randn_like(param) * 0.02)

    adapter_dir = TMP / "adapter"
    peft_model.save_pretrained(adapter_dir)
    adapter_mb = sum(f.stat().st_size for f in adapter_dir.iterdir()) / 1024 / 1024
    print(f"  adapter saved: {adapter_mb:.1f} MB")

    merged_dir = TMP / "merged"
    peft_model.merge_and_unload().save_pretrained(merged_dir, safe_serialization=True)
    print(f"  peft merge_and_unload() saved as the reference")

    print("\n" + "-" * 78)
    print("Import base and reference")
    print("-" * 78)
    base_db = str(TMP / "base.db")
    ref_db = str(TMP / "reference.db")
    safetensors_to_sqlite(str(MODEL_DIR), base_db, verbose=False)
    safetensors_to_sqlite(str(merged_dir), ref_db, verbose=False)
    print("  done")

    print("\n" + "-" * 78)
    print("Adapter -> delta pack")
    print("-" * 78)
    pack = str(TMP / "adapter.pack.db")
    summary = lora_to_delta_pack(str(adapter_dir), base_db, pack, verbose=True)

    print("\n" + "-" * 78)
    print("Apply, then compare against peft's merge")
    print("-" * 78)
    result = str(TMP / "result.db")
    apply_delta(base_db, pack, result, verify=True, verbose=True)

    res_conn = sqlite3.connect(result)
    ref_conn = sqlite3.connect(ref_db)

    worst, worst_name = 0.0, ""
    exact = 0
    total = 0
    max_ulp_gap = 0
    for name, dtype, blob in res_conn.execute("SELECT name, dtype, data FROM tensors"):
        row = ref_conn.execute("SELECT data FROM tensors WHERE name = ?", (name,)).fetchone()
        assert row is not None, f"{name} missing from peft's merged model"
        total += 1
        if blob == row[0]:
            exact += 1
            continue
        ours = to_float32(blob, dtype)
        theirs = to_float32(row[0], dtype)
        denom = np.linalg.norm(theirs)
        if denom > 0:
            rel = float(np.linalg.norm(ours - theirs) / denom)
            if rel > worst:
                worst, worst_name = rel, name
        # For BF16, adjacent representable values differ by one in the raw
        # 16-bit pattern, so this counts how many steps apart we are.
        gap = np.abs(
            np.frombuffer(blob, dtype=np.int16).astype(np.int32)
            - np.frombuffer(row[0], dtype=np.int16).astype(np.int32)
        ).max()
        max_ulp_gap = max(max_ulp_gap, int(gap))

    res_conn.close()
    ref_conn.close()

    print(f"\n  {total} tensors compared")
    print(f"  byte-identical to peft:        {exact}/{total}")
    print(f"  worst relative difference:     {worst:.3e} ({worst_name or 'none'})")
    print(f"  worst gap in BF16 steps (ULP): {max_ulp_gap}")

    pct = summary["pack_stored_bytes"] / summary["base_total_bytes"] * 100
    print(f"\n  base model: {summary['base_total_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  delta pack: {summary['pack_stored_bytes'] / 1024 / 1024:.1f} MB ({pct:.2f}% of base)")

    assert worst < 1e-3, (
        f"diverges from peft by {worst:.3e} at {worst_name} -- far more than "
        "BF16 rounding explains; check the scaling factor"
    )

    shutil.rmtree(TMP, ignore_errors=True)
    print("\n" + "=" * 78)
    print("REAL-MODEL LORA CHECK PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
