"""Verify a registry holds several models correctly and cheaply.

The point of a registry is that derived models cost their delta rather than a
full copy, and that they still come back out byte-exact. Both halves matter: a
compact store that reconstructs the wrong weights is worthless.
"""

import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reminis.converter import gguf_to_sqlite
from reminis.diff import _weights_hash
from reminis.registry import Registry

MODELS_DIR = Path(__file__).parent.parent / "models"
TMP = Path(__file__).parent / "tmp_registry"

BASE_GGUF = MODELS_DIR / "SmolLM-135M.f16.gguf"
INSTRUCT_GGUF = MODELS_DIR / "SmolLM-135M-Instruct.f16.gguf"


def build_lora_variant(db_path: str, seed: int, rank: int = 16) -> str:
    """A LoRA-shaped fine-tune of an existing model database."""
    import json

    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(db_path)
    targets = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")

    for name, shape_json, dtype, blob in conn.execute(
        "SELECT name, shape, dtype, data FROM tensors"
    ).fetchall():
        if not name.endswith(targets) or dtype != "F16":
            continue
        m, n = json.loads(shape_json)[::-1]
        w = np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(m, n)
        a = rng.normal(0, 1 / np.sqrt(rank), size=(rank, n)).astype(np.float32)
        b = rng.normal(0, 1 / np.sqrt(rank), size=(m, rank)).astype(np.float32)
        with np.errstate(all="ignore"):
            update = b @ a
        update *= 0.05 * np.linalg.norm(w) / np.linalg.norm(update)
        conn.execute(
            "UPDATE tensors SET data = ? WHERE name = ?",
            ((w + update).astype(np.float16).tobytes(), name),
        )
    conn.commit()
    conn.close()
    return db_path


def hash_of(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    h = _weights_hash(conn)
    conn.close()
    return h


def main():
    if not BASE_GGUF.exists():
        print(f"SKIP: {BASE_GGUF} not present")
        return

    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Multi-model registry")
    print("=" * 78)

    print("\nPreparing source models ...")
    base_db = str(TMP / "base.db")
    gguf_to_sqlite(str(BASE_GGUF), base_db, verbose=False)
    base_hash = hash_of(base_db)

    ft_a = build_lora_variant(shutil.copyfile(base_db, TMP / "ft_a.db"), seed=1)
    ft_b = build_lora_variant(shutil.copyfile(base_db, TMP / "ft_b.db"), seed=2)
    hashes = {"smollm": base_hash, "smollm-a": hash_of(ft_a), "smollm-b": hash_of(ft_b)}

    have_instruct = INSTRUCT_GGUF.exists()
    if have_instruct:
        inst_db = str(TMP / "instruct.db")
        gguf_to_sqlite(str(INSTRUCT_GGUF), inst_db, verbose=False)
        hashes["smollm-instruct"] = hash_of(inst_db)

    assert len(set(hashes.values())) == len(hashes), "fixtures are not distinct models"
    print(f"  {len(hashes)} distinct models prepared")

    registry_path = str(TMP / "registry.db")
    reg = Registry(registry_path)

    print("\n" + "-" * 78)
    print("Adding a base and three derived models")
    print("-" * 78)

    reg.add_base(base_db, "smollm", verbose=False)
    reg.add_derived(ft_a, "smollm-a", parent="smollm", verbose=False)
    reg.add_derived(ft_b, "smollm-b", parent="smollm", verbose=False)
    if have_instruct:
        reg.add_derived(inst_db, "smollm-instruct", parent="smollm", verbose=False)

    for m in reg.list_models():
        pct = m["stored_bytes"] / m["logical_bytes"] * 100 if m["logical_bytes"] else 0
        parent = f" from '{m['parent']}'" if m["parent"] else ""
        print(f"  {m['name']:<20} {m['kind']:<8}{parent:<18} "
              f"{m['logical_bytes'] / 1024 / 1024:8.1f} MB full -> "
              f"{m['stored_bytes'] / 1024 / 1024:7.1f} MB stored ({pct:.1f}%)")

    print("\n" + "-" * 78)
    print("Every model comes back byte-exact")
    print("-" * 78)
    for name, expected in hashes.items():
        actual = reg.weights_hash(name)
        status = "exact" if actual == expected else "MISMATCH"
        print(f"  {name:<20} {status}  ({actual[:16]}...)")
        assert actual == expected, f"{name} did not round-trip through the registry"

    print("\n" + "-" * 78)
    print("Materialising back to a standalone database")
    print("-" * 78)
    out = str(TMP / "out.db")
    reg.materialize("smollm-a", out, verify=True, verbose=False)
    assert hash_of(out) == hashes["smollm-a"]
    n_out = sqlite3.connect(out).execute("SELECT COUNT(*) FROM tensors").fetchone()[0]
    print(f"  smollm-a -> {n_out} tensors, hash verified")

    print("\n" + "-" * 78)
    print("A fine-tune of a fine-tune (chained deltas)")
    print("-" * 78)
    chained = build_lora_variant(shutil.copyfile(ft_a, TMP / "ft_a2.db"), seed=3)
    chained_hash = hash_of(chained)
    reg.add_derived(chained, "smollm-a2", parent="smollm-a", verbose=False)
    assert reg.weights_hash("smollm-a2") == chained_hash, \
        "a two-level delta chain did not resolve correctly"
    row = [m for m in reg.list_models() if m["name"] == "smollm-a2"][0]
    print(f"  smollm-a2 (from smollm-a, from smollm): "
          f"{row['stored_bytes'] / 1024 / 1024:.1f} MB stored, resolves exact")

    print("\n" + "-" * 78)
    print("What the registry saved")
    print("-" * 78)
    s = reg.stats()
    print(f"  models: {s['models']} ({s['bases']} base, {s['derived']} derived)")
    print(f"  stored separately: {s['logical_bytes'] / 1024 / 1024:8.1f} MB")
    print(f"  registry file:     {s['file_bytes'] / 1024 / 1024:8.1f} MB")
    print(f"  saved:             {s['savings'] * 100:.1f}%")
    assert s["file_bytes"] < s["logical_bytes"], "registry is not smaller than separate copies"

    print("\n" + "-" * 78)
    print("Guards")
    print("-" * 78)

    try:
        reg.add_base(base_db, "smollm", verbose=False)
    except ValueError as e:
        assert "already in this registry" in str(e)
        print("  duplicate name refused")
    else:
        raise AssertionError("expected a duplicate name to be refused")

    try:
        reg.remove("smollm")
    except ValueError as e:
        assert "parent of" in str(e)
        print("  removing a model that others derive from is refused")
    else:
        raise AssertionError("expected removal of a parent to be refused")

    try:
        reg._row("nope")
    except KeyError as e:
        assert "No model named" in str(e)
        print("  unknown model name reports what is available")

    reg.remove("smollm-a2", verbose=False)
    assert "smollm-a2" not in reg
    assert reg.weights_hash("smollm-a") == hashes["smollm-a"], \
        "removing a child disturbed its parent"
    print("  removing a leaf leaves its parent intact")
    reg.close()

    # Opening a single-model database as a registry must fail clearly rather
    # than grafting registry tables onto it.
    try:
        Registry(base_db)
    except ValueError as e:
        assert "single-model database" in str(e)
        print("  opening a single-model .db as a registry is refused")
    else:
        raise AssertionError("expected a single-model db to be rejected")

    # ...and that database must be untouched by the attempt.
    assert hash_of(base_db) == base_hash, "the rejected database was modified"
    print("  the rejected database was left unmodified")

    shutil.rmtree(TMP, ignore_errors=True)
    print("\n" + "=" * 78)
    print("ALL REGISTRY TESTS PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
