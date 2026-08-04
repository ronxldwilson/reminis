"""SHA256-verified lossless round-trip test for all GGUF quantization types."""

import hashlib
import time
from pathlib import Path

from gguf.gguf_reader import GGUFReader

from reminis.converter import gguf_to_sqlite, sqlite_to_gguf

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


def test_model(gguf_path: Path) -> dict:
    """Run full round-trip test with SHA256 verification on a single model."""
    name = gguf_path.stem
    db_path = str(TMP_DIR / f"{name}.db")
    rt_path = str(TMP_DIR / f"{name}.roundtrip.gguf")

    # Hash original
    t0 = time.time()
    orig_hashes = sha256_tensors(str(gguf_path))
    hash_time = time.time() - t0

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
        "mismatches": mismatches,
        "passed": len(mismatches) == 0,
    }


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
    if not all_files:
        print(f"No GGUF files found in {MODELS_DIR}")
        return

    models, skipped = [], []
    for p in all_files:
        (models if is_readable_gguf(p) else skipped).append(p)

    for p in skipped:
        mb = p.stat().st_size / (1024 * 1024)
        print(f"SKIP {p.name} ({mb:.0f} MB) -- not a complete GGUF, likely still downloading")
    if skipped:
        print()

    if not models:
        print("No complete GGUF files to test.")
        return

    print(f"Testing {len(models)} models with SHA256 verification\n")
    print(f"{'Model':<35} {'Dtypes':<20} {'Tensors':>8} {'GGUF MB':>8} {'DB MB':>8} {'RT MB':>8} {'Conv(s)':>8} {'Exp(s)':>8} {'Result':>8}")
    print("-" * 135)

    results = []
    all_passed = True

    for model_path in models:
        result = test_model(model_path)
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
