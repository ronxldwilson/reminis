"""SHA256-verified lossless round-trip test for every model in `models/`.

Covers GGUF (all quantization types) and safetensors (single-file, sharded,
and directory layouts). Both are discovered automatically, so dropping a model
into `models/` is enough to include it.
"""

import hashlib
import time
from pathlib import Path

from gguf.gguf_reader import GGUFReader

from reminis.converter import gguf_to_sqlite, sqlite_to_gguf
from reminis.safetensors_io import (
    read_header,
    resolve_shards,
    safetensors_to_sqlite,
    sqlite_to_safetensors,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
TMP_DIR = Path(__file__).parent / "tmp"


def sha256_tensors(gguf_path: str) -> dict[str, str]:
    """Compute SHA256 hash of every tensor's raw bytes in a GGUF file."""
    reader = GGUFReader(gguf_path, mode="r")
    hashes = {}
    for tensor in reader.tensors:
        raw = tensor.data.tobytes()
        hashes[tensor.name] = hashlib.sha256(raw).hexdigest()
    return hashes


def sha256_metadata(gguf_path: str) -> dict[str, str]:
    """Compute SHA256 hash of key metadata values."""
    reader = GGUFReader(gguf_path, mode="r")
    hashes = {}
    for key, field in reader.fields.items():
        if field.data:
            raw = field.parts[field.data[0]].tobytes()
            hashes[key] = hashlib.sha256(raw).hexdigest()
    return hashes


# GGUFReader synthesises these from the file header; they are not fields the
# file itself carries, so an export is not expected to reproduce them.
READER_ONLY_FIELDS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}


def metadata_fields(path: str) -> dict:
    """Every metadata field, as (rendered value, type chain).

    This suite compared tensors and nothing else, and so reported a lossless
    round-trip for files that had silently lost their tokenizer -- export
    dropped every array-typed field through 0.31.1. Weights matching is not
    the same as the model still loading.
    """
    from reminis.converter import _extract_field_value

    reader = GGUFReader(path, mode="r")
    return {
        key: (str(_extract_field_value(field)), list(field.types))
        for key, field in reader.fields.items()
        if key not in READER_ONLY_FIELDS
    }


def check_model(gguf_path: Path) -> dict:
    """Run full round-trip test with SHA256 verification on a single model.

    Named `check_` rather than `test_` on purpose: it takes a model file
    and returns a report, so pytest would collect it, find no fixture for
    `gguf_path`, and report a missing-fixture error that reads like a
    broken test. `main()` below is what supplies the models.
    """
    name = gguf_path.stem
    db_path = str(TMP_DIR / f"{name}.db")
    rt_path = str(TMP_DIR / f"{name}.roundtrip.gguf")

    # Hash original
    t0 = time.time()
    orig_hashes = sha256_tensors(str(gguf_path))
    hash_time = time.time() - t0
    orig_meta = metadata_fields(str(gguf_path))

    # Convert GGUF -> SQLite
    t0 = time.time()
    gguf_to_sqlite(str(gguf_path), db_path, verbose=False)
    convert_time = time.time() - t0

    # Convert SQLite -> GGUF
    t0 = time.time()
    sqlite_to_gguf(db_path, rt_path, verbose=False)
    export_time = time.time() - t0

    # Hash round-tripped
    t0 = time.time()
    rt_hashes = sha256_tensors(rt_path)
    verify_time = time.time() - t0

    # Compare metadata as well as tensors: a file whose weights match but
    # whose tokenizer was dropped is not a lossless round-trip.
    rt_meta = metadata_fields(rt_path)
    meta_mismatches = []
    for key in orig_meta:
        if key not in rt_meta:
            meta_mismatches.append((key, "META_MISSING"))
        elif orig_meta[key] != rt_meta[key]:
            meta_mismatches.append((key, "META_CHANGED"))

    # Compare
    mismatches = []
    for tensor_name in orig_hashes:
        if tensor_name not in rt_hashes:
            mismatches.append((tensor_name, "MISSING"))
        elif orig_hashes[tensor_name] != rt_hashes[tensor_name]:
            mismatches.append((tensor_name, "HASH_MISMATCH"))

    for tensor_name in rt_hashes:
        if tensor_name not in orig_hashes:
            mismatches.append((tensor_name, "EXTRA"))

    # Get dtype info
    reader = GGUFReader(str(gguf_path), mode="r")
    dtypes = set()
    for t in reader.tensors:
        dtypes.add(t.tensor_type.name)

    orig_size = gguf_path.stat().st_size / (1024 * 1024)
    db_size = Path(db_path).stat().st_size / (1024 * 1024)
    rt_size = Path(rt_path).stat().st_size / (1024 * 1024)

    return {
        "name": name,
        "dtypes": sorted(dtypes),
        "tensors": len(orig_hashes),
        "orig_mb": orig_size,
        "db_mb": db_size,
        "rt_mb": rt_size,
        "convert_s": convert_time,
        "export_s": export_time,
        "mismatches": mismatches + meta_mismatches,
        "meta_fields": len(orig_meta),
        "passed": len(mismatches) == 0 and len(meta_mismatches) == 0,
    }


def sha256_safetensors(path: Path) -> dict[str, str]:
    """SHA256 every tensor's raw bytes across a safetensors model's shards."""
    hashes = {}
    for shard in resolve_shards(path)[0]:
        header, data_start = read_header(shard)
        with open(shard, "rb") as f:
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                start, end = entry["data_offsets"]
                f.seek(data_start + start)
                hashes[name] = hashlib.sha256(f.read(end - start)).hexdigest()
    return hashes


def check_safetensors_model(path: Path) -> dict:
    """Round-trip a safetensors model through SQLite and verify every tensor."""
    name = path.name if path.is_dir() else path.stem
    db_path = str(TMP_DIR / f"{name}.st.db")
    out_dir = TMP_DIR / f"{name}-rt"
    out_dir.mkdir(parents=True, exist_ok=True)
    rt_path = out_dir / "model.safetensors"

    orig_hashes = sha256_safetensors(path)

    t0 = time.time()
    safetensors_to_sqlite(str(path), db_path, verbose=False)
    convert_time = time.time() - t0

    t0 = time.time()
    sqlite_to_safetensors(db_path, str(rt_path), verbose=False)
    export_time = time.time() - t0

    rt_hashes = sha256_safetensors(rt_path)

    mismatches = []
    for tensor_name in orig_hashes:
        if tensor_name not in rt_hashes:
            mismatches.append((tensor_name, "MISSING"))
        elif orig_hashes[tensor_name] != rt_hashes[tensor_name]:
            mismatches.append((tensor_name, "HASH_MISMATCH"))
    for tensor_name in rt_hashes:
        if tensor_name not in orig_hashes:
            mismatches.append((tensor_name, "EXTRA"))

    dtypes = set()
    for shard in resolve_shards(path)[0]:
        header, _ = read_header(shard)
        dtypes.update(
            e["dtype"] for n, e in header.items() if n != "__metadata__"
        )

    shards = resolve_shards(path)[0]
    orig_size = sum(s.stat().st_size for s in shards) / (1024 * 1024)

    return {
        "name": name,
        "dtypes": sorted(dtypes),
        "tensors": len(orig_hashes),
        "orig_mb": orig_size,
        "db_mb": Path(db_path).stat().st_size / (1024 * 1024),
        "rt_mb": rt_path.stat().st_size / (1024 * 1024),
        "convert_s": convert_time,
        "export_s": export_time,
        "mismatches": mismatches,
        "passed": len(mismatches) == 0,
    }


def is_readable_safetensors(path: Path) -> bool:
    """Cheap completeness check, matching the GGUF one.

    `models/` is a scratch area that can hold partial downloads. A truncated
    file has a valid header but its last tensor runs past the end, so check
    that the declared data block actually fits.
    """
    try:
        for shard in resolve_shards(path)[0]:
            header, data_start = read_header(shard)
            needed = max(
                e["data_offsets"][1]
                for name, e in header.items()
                if name != "__metadata__"
            )
            if shard.stat().st_size < data_start + needed:
                return False
        return True
    except Exception:
        return False


def find_safetensors_models() -> list[Path]:
    """Discover safetensors models: loose files, and directories holding them."""
    found = []
    for child in sorted(MODELS_DIR.iterdir()) if MODELS_DIR.exists() else []:
        if child.is_dir():
            if (child / "model.safetensors.index.json").exists():
                found.append(child / "model.safetensors.index.json")
            elif any(child.glob("*.safetensors")):
                found.append(child)
        elif child.suffix == ".safetensors":
            found.append(child)
    return found


def is_readable_gguf(path: Path) -> bool:
    """Cheap check that a file is a complete, parseable GGUF.

    The models directory is a working scratch area, so it can contain
    partially-downloaded files. Those should be skipped with a clear note
    rather than crashing the run or being reported as a real failure.
    """
    try:
        GGUFReader(str(path), mode="r")
        return True
    except Exception:
        return False


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    all_files = sorted(MODELS_DIR.glob("*.gguf"))

    st_models, st_skipped = [], []
    for p in find_safetensors_models():
        (st_models if is_readable_safetensors(p) else st_skipped).append(p)
    for p in st_skipped:
        print(f"SKIP {p.name} -- incomplete safetensors, likely still downloading")

    if not all_files and not st_models:
        print(f"No models found in {MODELS_DIR}")
        return

    models, skipped = [], []
    for p in all_files:
        (models if is_readable_gguf(p) else skipped).append(p)

    for p in skipped:
        mb = p.stat().st_size / (1024 * 1024)
        print(f"SKIP {p.name} ({mb:.0f} MB) -- not a complete GGUF, likely still downloading")
    if skipped:
        print()

    if not models and not st_models:
        print("No complete model files to test.")
        return

    print(f"Testing {len(models)} GGUF + {len(st_models)} safetensors models "
          f"with SHA256 verification\n")
    print(f"{'Model':<35} {'Dtypes':<20} {'Tensors':>8} {'GGUF MB':>8} {'DB MB':>8} {'RT MB':>8} {'Conv(s)':>8} {'Exp(s)':>8} {'Result':>8}")
    print("-" * 135)

    results = []
    all_passed = True

    for model_path in models:
        result = check_model(model_path)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        if not result["passed"]:
            all_passed = False

        dtypes_str = ",".join(result["dtypes"])
        print(
            f"{result['name']:<35} {dtypes_str:<20} {result['tensors']:>8} "
            f"{result['orig_mb']:>8.1f} {result['db_mb']:>8.1f} {result['rt_mb']:>8.1f} "
            f"{result['convert_s']:>8.2f} {result['export_s']:>8.2f} {status:>8}"
        )

        if not result["passed"]:
            for tensor_name, error in result["mismatches"][:5]:
                print(f"  !! {tensor_name}: {error}")

    for st_path in st_models:
        result = check_safetensors_model(st_path)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        if not result["passed"]:
            all_passed = False

        dtypes_str = ",".join(result["dtypes"])
        print(
            f"{result['name']:<35} {dtypes_str:<20} {result['tensors']:>8} "
            f"{result['orig_mb']:>8.1f} {result['db_mb']:>8.1f} {result['rt_mb']:>8.1f} "
            f"{result['convert_s']:>8.2f} {result['export_s']:>8.2f} {status:>8}"
        )

        if not result["passed"]:
            for tensor_name, error in result["mismatches"][:5]:
                print(f"  !! {tensor_name}: {error}")

    print("-" * 135)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{passed}/{len(results)} models passed SHA256-verified lossless round-trip")

    if all_passed:
        print("\nALL TESTS PASSED - every tensor in every model matches byte-for-byte")
    else:
        print("\nSOME TESTS FAILED")
        exit(1)

    # Cleanup
    import shutil
    shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    main()
