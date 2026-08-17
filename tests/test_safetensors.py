"""Verify safetensors import/export and the BF16 numeric path.

Fixtures are written by the real `safetensors` library (and torch, for BF16)
rather than by reminis' own writer, so this checks interoperability rather than
self-consistency. Those two packages are test-only; reminis itself parses the
format with nothing but numpy.
"""

import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reminis.dtypes import from_float32, to_float32
from reminis.safetensors_io import (
    read_header,
    safetensors_to_sqlite,
    sqlite_to_safetensors,
)

TMP = Path(__file__).parent / "tmp_safetensors"

try:
    import torch
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover - fixtures need the real libraries
    print("SKIP: install `torch` and `safetensors` to run this test")
    sys.exit(0)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_bf16_matches_torch():
    """reminis' BF16 conversion must agree with torch bit for bit."""
    print("-" * 78)
    print("BF16 conversion vs torch")
    print("-" * 78)

    rng = np.random.default_rng(0)
    values = np.concatenate([
        rng.normal(0, 1, 4096).astype(np.float32),
        rng.normal(0, 1e-8, 512).astype(np.float32),
        np.array([0.0, -0.0, 1.0, -1.0, 65504.0, 1e30, -1e30,
                  np.inf, -np.inf], dtype=np.float32),
    ])

    # f32 -> bf16: compare raw bits against torch's own rounding.
    ours = from_float32(values, "BF16")
    theirs = torch.tensor(values).to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    assert ours == theirs, "BF16 narrowing disagrees with torch"
    print(f"  f32 -> bf16: {len(values)} values, bit-identical to torch")

    # bf16 -> f32: the widening direction.
    back = to_float32(ours, "BF16")
    torch_back = torch.frombuffer(bytearray(ours), dtype=torch.bfloat16).float().numpy()
    assert np.array_equal(back, torch_back, equal_nan=True), "BF16 widening disagrees"
    print(f"  bf16 -> f32: matches torch exactly")

    # And the reason this matters: reading BF16 bytes as float16 is silent
    # garbage, which is what reminis did before this path existed.
    wrong = np.frombuffer(ours, dtype=np.float16).astype(np.float32)
    assert not np.allclose(wrong[:100], back[:100]), "expected float16 misread to differ"
    print("  (confirmed: misreading BF16 as float16 produces different values)")


def build_model(directory: Path, shard: bool = False) -> dict:
    """Write a small multi-dtype model with the real safetensors library."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    tensors = {
        "model.embed_tokens.weight": torch.randn(64, 32, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32).to(torch.bfloat16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(32, 32).to(torch.bfloat16),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(32, 32).to(torch.bfloat16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(32, 32).to(torch.bfloat16),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(64, 32).to(torch.float16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(32, 64).to(torch.float16),
        "model.layers.0.input_layernorm.weight": torch.randn(32, dtype=torch.float32),
        "model.norm.weight": torch.randn(32, dtype=torch.float32),
        "model.position_ids": torch.arange(32, dtype=torch.int64),
    }

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "torch_dtype": "bfloat16",
        "vocab_size": 64,
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2))

    if shard:
        names = list(tensors)
        halves = [names[: len(names) // 2], names[len(names) // 2 :]]
        weight_map = {}
        for i, group in enumerate(halves, start=1):
            filename = f"model-{i:05d}-of-{len(halves):05d}.safetensors"
            save_file({k: tensors[k] for k in group}, directory / filename,
                      metadata={"format": "pt"})
            for k in group:
                weight_map[k] = filename
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 12345}, "weight_map": weight_map})
        )
    else:
        save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})

    return tensors


def reference_bytes(tensors: dict) -> dict:
    """The raw bytes torch holds for each tensor, for hash comparison."""
    out = {}
    for name, t in tensors.items():
        out[name] = t.contiguous().view(torch.uint8).numpy().tobytes() \
            if t.dtype == torch.bfloat16 else t.numpy().tobytes()
    return out


def test_roundtrip(shard: bool):
    label = "sharded" if shard else "single-file"
    print("\n" + "-" * 78)
    print(f"Round-trip: {label} safetensors -> SQLite -> safetensors")
    print("-" * 78)

    directory = TMP / ("sharded" if shard else "single")
    shutil.rmtree(directory, ignore_errors=True)
    tensors = build_model(directory, shard=shard)
    expected = reference_bytes(tensors)

    db = str(TMP / f"{label}.db")
    safetensors_to_sqlite(str(directory), db, verbose=False)

    conn = sqlite3.connect(db)
    rows = dict(
        (n, (s, d, blob))
        for n, s, d, blob in conn.execute("SELECT name, shape, dtype, data FROM tensors")
    )
    assert set(rows) == set(tensors), (
        f"tensor set differs: missing {set(tensors) - set(rows)}, "
        f"extra {set(rows) - set(tensors)}"
    )

    for name, torch_tensor in tensors.items():
        shape_json, dtype, blob = rows[name]
        assert sha(blob) == sha(expected[name]), f"bytes differ for {name}"
        # Shape is stored reversed relative to the data layout.
        assert json.loads(shape_json) == list(torch_tensor.shape)[::-1], \
            f"shape stored wrong for {name}"
    print(f"  {len(tensors)} tensors imported, all SHA256-identical")

    dtypes = {r[0] for r in conn.execute("SELECT DISTINCT dtype FROM tensors")}
    assert "BF16" in dtypes, "BF16 tensors were not preserved as BF16"
    print(f"  dtypes preserved: {', '.join(sorted(dtypes))}")

    config_keys = [
        r[0] for r in conn.execute("SELECT key FROM model_meta WHERE key LIKE 'config.%'")
    ]
    assert "config.hidden_size" in config_keys, "config.json was not ingested"
    arch = conn.execute(
        "SELECT value FROM model_meta WHERE key = 'general.architecture'"
    ).fetchone()
    assert arch and arch[0] == "LlamaForCausalLM"
    print(f"  config.json ingested ({len(config_keys)} keys), architecture={arch[0]}")
    conn.close()

    # Export and re-read.
    out_dir = TMP / f"{label}-export"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / "model.safetensors")
    sqlite_to_safetensors(db, out_file, verbose=False)

    from safetensors import safe_open

    with safe_open(out_file, framework="pt") as f:
        exported = set(f.keys())
    assert exported == set(tensors), "exported tensor set differs"

    header, data_start = read_header(Path(out_file))
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        with open(out_file, "rb") as f:
            f.seek(data_start + start)
            blob = f.read(end - start)
        assert sha(blob) == sha(expected[name]), f"exported bytes differ for {name}"
        assert entry["shape"] == list(tensors[name].shape), f"exported shape wrong for {name}"
    print(f"  exported file re-reads identically (verified with the real library)")

    assert (out_dir / "config.json").exists(), "config.json was not written on export"
    written = json.loads((out_dir / "config.json").read_text())
    assert written["hidden_size"] == 32 and written["architectures"] == ["LlamaForCausalLM"]
    print("  config.json rebuilt on export with types intact")


def test_gguf_export_guard():
    """A quantized GGUF model cannot become safetensors, and must say so."""
    print("\n" + "-" * 78)
    print("Guard: quantized GGUF -> safetensors is refused")
    print("-" * 78)

    quantized = Path(__file__).parent.parent / "models" / "granite-3.1-1b-a400m-instruct-Q4_K_M.db"
    if not quantized.exists():
        print("  SKIP: no quantized model database available")
        return
    try:
        sqlite_to_safetensors(str(quantized), str(TMP / "nope.safetensors"), verbose=False)
    except ValueError as e:
        assert "no safetensors equivalent" in str(e), f"unexpected message: {e}"
        print(f"  refused as expected: {str(e).splitlines()[0][:70]}...")
        return
    raise AssertionError("quantized export should have been refused")


def test_gguf_db_exports_to_safetensors():
    """An F16 GGUF model has only safetensors-compatible dtypes, so it converts."""
    print("\n" + "-" * 78)
    print("Cross-format: F16 GGUF database -> safetensors")
    print("-" * 78)

    src = Path(__file__).parent.parent / "models" / "SmolLM-135M.f16.db"
    if not src.exists():
        print("  SKIP: SmolLM-135M.f16.db not present")
        return

    out = TMP / "from-gguf"
    out.mkdir(parents=True, exist_ok=True)
    out_file = str(out / "model.safetensors")
    sqlite_to_safetensors(str(src), out_file, verbose=False)

    conn = sqlite3.connect(str(src))
    header, data_start = read_header(Path(out_file))
    checked = 0
    for name, shape_json, blob in conn.execute("SELECT name, shape, data FROM tensors"):
        entry = header[name]
        start, end = entry["data_offsets"]
        with open(out_file, "rb") as f:
            f.seek(data_start + start)
            assert sha(f.read(end - start)) == sha(blob), f"bytes differ for {name}"
        assert entry["shape"] == json.loads(shape_json)[::-1]
        checked += 1
    conn.close()
    print(f"  {checked} tensors exported byte-identically, shapes un-reversed")
