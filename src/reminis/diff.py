"""Compare two model databases and produce reusable delta packs.

A delta pack stores only what changed between a base model and a target model.
Applying a pack to the base reconstructs the target, so a fine-tune can be
distributed as a small migration instead of a full model copy.
"""

import hashlib
import json
import sqlite3
import time
import zlib
from pathlib import Path

import numpy as np

from gguf.constants import GGMLQuantizationType

# Tensor types whose bytes decode directly to floats, so deltas are meaningful.
NUMERIC_DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.float16,
}

DELTA_SCHEMA = """
CREATE TABLE IF NOT EXISTS delta_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS deltas (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tensor_name   TEXT UNIQUE NOT NULL,
    shape         TEXT NOT NULL,
    dtype         TEXT NOT NULL,
    dtype_id      INTEGER NOT NULL,
    n_elements    INTEGER NOT NULL,
    n_changed     INTEGER,
    pct_changed   REAL,
    l2_norm       REAL,
    max_abs_delta REAL,
    mean_delta    REAL,
    encoding      TEXT NOT NULL,
    raw_bytes     INTEGER NOT NULL,
    stored_bytes  INTEGER NOT NULL,
    data          BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_deltas_name ON deltas(tensor_name);
"""


def _model_fingerprint(conn: sqlite3.Connection) -> str:
    """Hash tensor names, shapes, and dtypes to identify a model's structure.

    Deliberately excludes weight data so the hash stays cheap on large models.
    Two models with the same fingerprint are delta-compatible.
    """
    h = hashlib.sha256()
    for name, shape, dtype in conn.execute(
        "SELECT name, shape, dtype FROM tensors ORDER BY name"
    ):
        h.update(f"{name}|{shape}|{dtype}".encode())
    return h.hexdigest()


def _weights_hash(conn: sqlite3.Connection) -> str:
    """Hash the actual weight data. Identifies an exact model state."""
    h = hashlib.sha256()
    for (blob,) in conn.execute("SELECT data FROM tensors ORDER BY name"):
        h.update(blob)
    return h.hexdigest()


def _tensor_delta_stats(a_blob: bytes, b_blob: bytes, dtype: str):
    """Compute change statistics between two tensors of the same dtype.

    Returns (stats_dict, delta_array_or_None). The delta array is None for
    quantized types, where raw bytes cannot be subtracted meaningfully.
    """
    if a_blob == b_blob:
        return {"identical": True}, None

    np_dtype = NUMERIC_DTYPES.get(dtype)
    if np_dtype is None:
        # Quantized: we can tell that it changed, but not by how much.
        return {
            "identical": False,
            "numeric": False,
            "n_changed": None,
            "pct_changed": None,
            "l2_norm": None,
            "max_abs_delta": None,
            "mean_delta": None,
        }, None

    a = np.frombuffer(a_blob, dtype=np_dtype).astype(np.float32)
    b = np.frombuffer(b_blob, dtype=np_dtype).astype(np.float32)

    if a.shape != b.shape:
        return {"identical": False, "numeric": False, "shape_mismatch": True}, None

    delta = b - a
    changed_mask = delta != 0
    n_changed = int(np.count_nonzero(changed_mask))

    return {
        "identical": False,
        "numeric": True,
        "n_changed": n_changed,
        "pct_changed": float(n_changed / len(delta) * 100),
        "l2_norm": float(np.linalg.norm(delta)),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "mean_delta": float(np.mean(delta)),
    }, delta


def _encode_delta(a_blob: bytes, b_blob: bytes) -> tuple[str, bytes]:
    """Pick the smaller of a compressed XOR delta or a compressed replacement.

    XOR is used rather than arithmetic subtraction because it is exactly
    reversible for every dtype. An arithmetic float delta is not: `b - a`
    generally is not representable in the tensor's own dtype, so `a + delta`
    lands a rounding step away from `b`. XOR also works on quantized tensors,
    whose bytes cannot be subtracted meaningfully at all.

    Bytes that did not change XOR to zero, so runs of unchanged data compress
    away and the payload tracks how much actually moved.

    Returns (encoding_name, payload). 'xor_zlib' payloads are XORed against the
    base tensor on apply; 'replace_zlib' payloads overwrite it outright.
    """
    replacement = zlib.compress(b_blob, level=6)

    if len(a_blob) != len(b_blob):
        return "replace_zlib", replacement

    a_arr = np.frombuffer(a_blob, dtype=np.uint8)
    b_arr = np.frombuffer(b_blob, dtype=np.uint8)
    xor_payload = zlib.compress(np.bitwise_xor(a_arr, b_arr).tobytes(), level=6)

    if len(xor_payload) <= len(replacement):
        return "xor_zlib", xor_payload
    return "replace_zlib", replacement


def diff_models(
    db_a: str,
    db_b: str,
    output_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """Compare two model databases, optionally writing a delta pack.

    Args:
        db_a: Path to the base model database.
        db_b: Path to the target model database.
        output_path: If given, write a delta pack that turns A into B.
        verbose: Print a progress report.

    Returns:
        A summary dict describing what changed.
    """
    path_a, path_b = Path(db_a), Path(db_b)
    for p in (path_a, path_b):
        if not p.exists():
            raise FileNotFoundError(f"Database not found: {p}")

    t0 = time.time()
    conn_a = sqlite3.connect(str(path_a))
    conn_b = sqlite3.connect(str(path_b))

    fp_a = _model_fingerprint(conn_a)
    fp_b = _model_fingerprint(conn_b)

    names_a = {r[0] for r in conn_a.execute("SELECT name FROM tensors")}
    names_b = {r[0] for r in conn_b.execute("SELECT name FROM tensors")}

    only_in_a = sorted(names_a - names_b)
    only_in_b = sorted(names_b - names_a)
    shared = sorted(names_a & names_b)

    if verbose:
        print(f"Comparing {path_a.name} -> {path_b.name}")
        if fp_a != fp_b:
            print("  Note: model structures differ (different tensor names/shapes/dtypes)")
        print(f"  {len(shared)} shared tensors, {len(only_in_a)} removed, {len(only_in_b)} added\n")

    out_conn = None
    if output_path:
        Path(output_path).unlink(missing_ok=True)
        out_conn = sqlite3.connect(output_path)
        out_conn.execute("PRAGMA journal_mode=WAL")
        out_conn.execute("PRAGMA synchronous=NORMAL")
        out_conn.executescript(DELTA_SCHEMA)

    changed = []
    identical_count = 0
    quantized_changed = 0
    total_raw_bytes = 0
    total_stored_bytes = 0

    for name in shared:
        row_a = conn_a.execute(
            "SELECT shape, dtype, dtype_id, n_elements, data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        row_b = conn_b.execute(
            "SELECT shape, dtype, dtype_id, n_elements, data FROM tensors WHERE name = ?", (name,)
        ).fetchone()

        shape_a, dtype_a, dtype_id_a, n_elements_a, blob_a = row_a
        shape_b, dtype_b, dtype_id_b, n_elements_b, blob_b = row_b

        if dtype_a != dtype_b or shape_a != shape_b:
            stats = {"identical": False, "numeric": False, "incompatible": True}
            delta = None
        else:
            stats, delta = _tensor_delta_stats(blob_a, blob_b, dtype_a)

        if stats.get("identical"):
            identical_count += 1
            continue

        if not stats.get("numeric"):
            quantized_changed += 1

        entry = {"name": name, "dtype": dtype_b, "shape": json.loads(shape_b), **stats}
        changed.append(entry)

        if out_conn is not None:
            encoding, payload = _encode_delta(blob_a, blob_b)
            total_raw_bytes += len(blob_b)
            total_stored_bytes += len(payload)

            out_conn.execute(
                "INSERT INTO deltas (tensor_name, shape, dtype, dtype_id, n_elements, "
                "n_changed, pct_changed, l2_norm, max_abs_delta, mean_delta, "
                "encoding, raw_bytes, stored_bytes, data) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    name, shape_b, dtype_b, dtype_id_b, n_elements_b,
                    stats.get("n_changed"), stats.get("pct_changed"),
                    stats.get("l2_norm"), stats.get("max_abs_delta"), stats.get("mean_delta"),
                    encoding, len(blob_b), len(payload), payload,
                ),
            )

    # Tensors present only in the target must ship in full.
    for name in only_in_b:
        row = conn_b.execute(
            "SELECT shape, dtype, dtype_id, n_elements, data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        shape, dtype, dtype_id, n_elements, blob = row
        if out_conn is not None:
            payload = zlib.compress(blob, level=6)
            total_raw_bytes += len(blob)
            total_stored_bytes += len(payload)
            out_conn.execute(
                "INSERT INTO deltas (tensor_name, shape, dtype, dtype_id, n_elements, "
                "encoding, raw_bytes, stored_bytes, data) VALUES (?,?,?,?,?,?,?,?)",
                (name, shape, dtype, dtype_id, n_elements, "replace_zlib", len(blob), len(payload), payload),
            )

    total_b_bytes = conn_b.execute("SELECT SUM(n_bytes) FROM tensors").fetchone()[0] or 0

    if out_conn is not None:
        meta = {
            "base_fingerprint": fp_a,
            "target_fingerprint": fp_b,
            "base_weights_hash": _weights_hash(conn_a),
            "target_weights_hash": _weights_hash(conn_b),
            "base_file": path_a.name,
            "target_file": path_b.name,
            "tensors_changed": str(len(changed)),
            "tensors_identical": str(identical_count),
            "tensors_removed": json.dumps(only_in_a),
            "reminis_version": _version(),
        }
        for k, v in meta.items():
            out_conn.execute("INSERT OR REPLACE INTO delta_meta (key, value) VALUES (?,?)", (k, v))
        out_conn.commit()
        out_conn.close()

    conn_a.close()
    conn_b.close()

    summary = {
        "base": str(path_a),
        "target": str(path_b),
        "shared": len(shared),
        "identical": identical_count,
        "changed": len(changed),
        "only_in_base": only_in_a,
        "only_in_target": only_in_b,
        "quantized_changed": quantized_changed,
        "changed_tensors": changed,
        "target_total_bytes": total_b_bytes,
        "delta_raw_bytes": total_raw_bytes,
        "delta_stored_bytes": total_stored_bytes,
        "elapsed": time.time() - t0,
    }

    if verbose:
        _print_diff_report(summary, output_path)

    return summary


def _version() -> str:
    from reminis import __version__
    return __version__


def _fmt_bytes(b: int) -> str:
    if b >= 1024 ** 3:
        return f"{b / 1024 ** 3:.1f} GB"
    if b >= 1024 ** 2:
        return f"{b / 1024 ** 2:.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def _print_diff_report(s: dict, output_path: str | None):
    print(f"  Identical: {s['identical']}")
    print(f"  Changed:   {s['changed']}")
    if s["only_in_base"]:
        print(f"  Removed:   {len(s['only_in_base'])}")
    if s["only_in_target"]:
        print(f"  Added:     {len(s['only_in_target'])}")

    numeric = [t for t in s["changed_tensors"] if t.get("numeric")]
    if numeric:
        top = sorted(numeric, key=lambda t: t["l2_norm"], reverse=True)[:10]
        print(f"\n  Top {len(top)} most-changed tensors (by L2 norm of the delta):")
        for t in top:
            print(
                f"    {t['name']:<40s} L2={t['l2_norm']:>10.3f}  "
                f"max|d|={t['max_abs_delta']:>8.4f}  {t['pct_changed']:>5.1f}% of values"
            )

    if s["quantized_changed"]:
        print(
            f"\n  {s['quantized_changed']} changed tensors are quantized -- "
            "detected as different, but per-value deltas are not computable."
        )

    if output_path:
        stored = s["delta_stored_bytes"]
        full = s["target_total_bytes"]
        print(f"\n  Delta pack: {output_path}")
        print(f"    Full model:  {_fmt_bytes(full)}")
        print(f"    Delta pack:  {_fmt_bytes(stored)}")
        if full:
            print(f"    Ratio:       {stored / full * 100:.1f}% of full model size")

    print(f"\n  Took {s['elapsed']:.1f}s")


def apply_delta(
    base_db: str,
    delta_db: str,
    output_db: str,
    verify: bool = True,
    verbose: bool = True,
) -> str:
    """Apply a delta pack to a base model, producing the target model.

    Args:
        base_db: Path to the base model database.
        delta_db: Path to the delta pack.
        output_db: Path for the resulting model database.
        verify: Check that the base matches what the pack was built against,
            and that the result matches the recorded target hash.
        verbose: Print progress.

    Returns:
        Path to the resulting database.
    """
    base_path, delta_path = Path(base_db), Path(delta_db)
    for p in (base_path, delta_path):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    t0 = time.time()
    Path(output_db).unlink(missing_ok=True)

    # Start from a copy of the base so untouched tensors carry over.
    import shutil
    shutil.copyfile(base_path, output_db)

    out_conn = sqlite3.connect(output_db)
    delta_conn = sqlite3.connect(str(delta_path))

    meta = dict(delta_conn.execute("SELECT key, value FROM delta_meta"))

    if verify:
        actual = _weights_hash(out_conn)
        expected = meta.get("base_weights_hash")
        if expected and actual != expected:
            out_conn.close()
            delta_conn.close()
            Path(output_db).unlink(missing_ok=True)
            raise ValueError(
                "Base model does not match the one this delta pack was built from.\n"
                f"  expected weights hash: {expected[:16]}...\n"
                f"  actual weights hash:   {actual[:16]}...\n"
                "Applying it would produce a corrupt model."
            )
        if verbose:
            print(f"Base verified against pack ({meta.get('base_file', 'unknown')})")

    # Drop tensors the target no longer has.
    removed = json.loads(meta.get("tensors_removed", "[]"))
    for name in removed:
        out_conn.execute("DELETE FROM tensors WHERE name = ?", (name,))

    applied = 0
    rows = delta_conn.execute(
        "SELECT tensor_name, shape, dtype, dtype_id, n_elements, encoding, data FROM deltas"
    ).fetchall()

    for name, shape, dtype, dtype_id, n_elements, encoding, payload in rows:
        raw = zlib.decompress(payload)

        if encoding == "xor_zlib":
            existing = out_conn.execute(
                "SELECT data FROM tensors WHERE name = ?", (name,)
            ).fetchone()
            if existing is None:
                raise ValueError(f"Delta references tensor '{name}' missing from the base model")
            base_arr = np.frombuffer(existing[0], dtype=np.uint8)
            delta_arr = np.frombuffer(raw, dtype=np.uint8)
            if len(base_arr) != len(delta_arr):
                raise ValueError(
                    f"Size mismatch applying delta to '{name}': "
                    f"base is {len(base_arr)} bytes, delta expects {len(delta_arr)}"
                )
            new_blob = np.bitwise_xor(base_arr, delta_arr).tobytes()
        else:
            new_blob = raw

        out_conn.execute(
            "INSERT INTO tensors (name, shape, dtype, dtype_id, n_elements, n_bytes, data) "
            "VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET shape=excluded.shape, dtype=excluded.dtype, "
            "dtype_id=excluded.dtype_id, n_elements=excluded.n_elements, "
            "n_bytes=excluded.n_bytes, data=excluded.data",
            (name, shape, dtype, dtype_id, n_elements, len(new_blob), new_blob),
        )
        applied += 1

    out_conn.commit()
    delta_conn.close()

    if verify:
        expected = meta.get("target_weights_hash")
        if expected:
            actual = _weights_hash(out_conn)
            if actual != expected:
                out_conn.close()
                raise ValueError(
                    "Result does not match the pack's recorded target hash.\n"
                    f"  expected: {expected[:16]}...\n"
                    f"  actual:   {actual[:16]}..."
                )
            if verbose:
                print("Result verified against target hash")

    out_conn.close()

    if verbose:
        print(f"Applied {applied} tensor updates -> {output_db}")
        print(f"Took {time.time() - t0:.1f}s")

    return output_db
