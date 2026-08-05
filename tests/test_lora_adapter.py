"""Verify that a peft LoRA adapter converts to a delta pack that merges correctly.

The fixture is a real peft adapter trained-shaped by peft itself, and the
reference answer comes from peft's own `merge_and_unload()`. That matters more
than the round-trip: the failure mode for this path is not a crash but a delta
of the wrong magnitude (a mis-applied `alpha / r`), which would look like a
slightly worse model rather than a bug.

peft, transformers, and torch are test-only. reminis reads the adapter with
numpy alone.
"""

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reminis.diff import _weights_hash, apply_delta
from reminis.dtypes import to_float32
from reminis.lora import lora_to_delta_pack
from reminis.safetensors_io import safetensors_to_sqlite

TMP = Path(__file__).parent / "tmp_lora_adapter"

RANK = 8
ALPHA = 16

try:
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:
    print("SKIP: install `torch`, `transformers`, and `peft` to run this test")
    sys.exit(0)


def build_base_and_adapter(directory: Path):
    """Create a tiny Llama, save it, and attach a randomised LoRA adapter."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = LlamaForCausalLM(config)
    model = model.to(torch.float32).eval()

    base_dir = directory / "base"
    model.save_pretrained(base_dir, safe_serialization=True)

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    # peft initialises lora_B to zeros so an untrained adapter is a no-op.
    # Fill both factors so the merge actually moves the weights -- otherwise
    # this test would pass with the scaling factor completely wrong.
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.copy_(torch.randn_like(param) * 0.05)

    adapter_dir = directory / "adapter"
    peft_model.save_pretrained(adapter_dir)

    # peft's own answer for what the merged model should be.
    merged = peft_model.merge_and_unload()
    merged_dir = directory / "merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)

    return base_dir, adapter_dir, merged_dir


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"LoRA adapter -> delta pack (rank {RANK}, alpha {ALPHA})")
    print("=" * 78)

    base_dir, adapter_dir, merged_dir = build_base_and_adapter(TMP)

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    print(f"\n  peft wrote r={adapter_config['r']}, "
          f"lora_alpha={adapter_config['lora_alpha']}, "
          f"scaling={adapter_config['lora_alpha'] / adapter_config['r']}")

    base_db = str(TMP / "base.db")
    merged_db = str(TMP / "merged_reference.db")
    safetensors_to_sqlite(str(base_dir), base_db, verbose=False)
    safetensors_to_sqlite(str(merged_dir), merged_db, verbose=False)
    print(f"  base and peft-merged reference imported")

    # A sanity check on the fixture itself: the adapter must actually change
    # something, or every assertion below would pass vacuously.
    base_conn = sqlite3.connect(base_db)
    ref_conn = sqlite3.connect(merged_db)
    assert _weights_hash(base_conn) != _weights_hash(ref_conn), \
        "peft's merge changed nothing; the fixture is broken"
    base_conn.close()
    ref_conn.close()

    print("\n" + "-" * 78)
    print("Convert the adapter")
    print("-" * 78)
    pack = str(TMP / "adapter.pack.db")
    summary = lora_to_delta_pack(str(adapter_dir), base_db, pack, verbose=True)

    assert summary["lowrank_tensors"] == 8, \
        f"expected 8 targeted modules (4 per layer x 2 layers), got {summary['lowrank_tensors']}"
    assert summary["ranks"] == [RANK], f"unexpected ranks {summary['ranks']}"

    print("\n" + "-" * 78)
    print("Apply the pack")
    print("-" * 78)
    result = str(TMP / "result.db")
    apply_delta(base_db, pack, result, verify=True, verbose=True)

    print("\n" + "-" * 78)
    print("Compare against peft's own merge_and_unload()")
    print("-" * 78)

    res_conn = sqlite3.connect(result)
    ref_conn = sqlite3.connect(merged_db)

    worst = 0.0
    worst_name = ""
    exact = 0
    compared = 0
    for name, dtype, blob in res_conn.execute("SELECT name, dtype, data FROM tensors"):
        row = ref_conn.execute(
            "SELECT data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        assert row is not None, f"{name} missing from peft's merged model"
        if blob == row[0]:
            exact += 1
        ours = to_float32(blob, dtype)
        theirs = to_float32(row[0], dtype)
        denom = np.linalg.norm(theirs)
        if denom > 0:
            rel = float(np.linalg.norm(ours - theirs) / denom)
            if rel > worst:
                worst, worst_name = rel, name
        compared += 1

    total = res_conn.execute("SELECT COUNT(*) FROM tensors").fetchone()[0]
    res_conn.close()
    ref_conn.close()

    print(f"  {compared} tensors compared, {exact} byte-identical")
    print(f"  worst relative difference: {worst:.3e}  ({worst_name or 'none'})")

    # A wrong alpha/r would show up around 1e0, not 1e-7, so this bound is
    # about catching a scaling bug rather than about numerical taste.
    assert worst < 1e-6, (
        f"merged result diverges from peft by {worst:.3e} at {worst_name}; "
        "this is far too large to be summation order -- check the scaling factor"
    )
    assert exact >= total - summary["lowrank_tensors"], \
        "tensors the adapter does not target should be untouched"

    if exact == compared:
        # Observed on this fixture: float32 base, float32 factors, and the same
        # arithmetic peft performs, so nothing rounds differently. Not promised
        # in general -- a bf16 base would round the final store.
        print("\n  Every tensor is byte-identical to peft's merge, including the 8 it modified.")
    else:
        print(f"\n  Matches peft's merge to within floating-point summation order.")

    print("\n" + "-" * 78)
    print("Size")
    print("-" * 78)
    pct = summary["pack_stored_bytes"] / summary["base_total_bytes"] * 100
    print(f"  base model: {summary['base_total_bytes'] / 1024:.1f} KB")
    print(f"  delta pack: {summary['pack_stored_bytes'] / 1024:.1f} KB ({pct:.1f}%)")
    print("  (a toy model's ratio is not meaningful -- rank 8 factors against a")
    print("   32-wide hidden size is nearly the full matrix. The ratio is worth")
    print("   reading only on a real model.)")

    print("\n" + "-" * 78)
    print("Guard: applying to the wrong base is rejected")
    print("-" * 78)
    wrong_base = str(TMP / "wrong.db")
    shutil.copyfile(base_db, wrong_base)
    conn = sqlite3.connect(wrong_base)
    name, blob = conn.execute(
        "SELECT name, data FROM tensors LIMIT 1"
    ).fetchone()
    tampered = bytearray(blob)
    tampered[0] ^= 0xFF
    conn.execute("UPDATE tensors SET data = ? WHERE name = ?", (bytes(tampered), name))
    conn.commit()
    conn.close()

    try:
        apply_delta(wrong_base, pack, str(TMP / "bad.db"), verify=True, verbose=False)
    except ValueError as e:
        assert "does not match" in str(e)
        print(f"  rejected as expected")
    else:
        raise AssertionError("applying to a tampered base should have been rejected")

    shutil.rmtree(TMP, ignore_errors=True)
    print("\n" + "=" * 78)
    print("ALL LORA ADAPTER TESTS PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
