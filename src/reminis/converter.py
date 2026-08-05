"""Convert between model files and SQLite databases.

GGUF lives here; safetensors lives in ``safetensors_io`` and shares this
schema. ``shape`` is stored reversed relative to the data layout for every
format, because that is GGUF's convention and the diff, low-rank, and viewer
code all read it that way.
"""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from gguf.gguf_reader import GGUFReader
from gguf import GGUFWriter
from gguf.constants import GGMLQuantizationType, GGUFValueType, GGML_QUANT_SIZES

from reminis.dtypes import DTYPE_SYSTEM_GGUF, DTYPE_SYSTEM_SAFETENSORS, dtype_system


SCHEMA = """
CREATE TABLE IF NOT EXISTS model_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    type  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tensors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE NOT NULL,
    shape      TEXT NOT NULL,
    dtype      TEXT NOT NULL,
    dtype_id   INTEGER NOT NULL,
    n_elements INTEGER NOT NULL,
    n_bytes    INTEGER NOT NULL,
    data       BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tensors_name ON tensors(name);
CREATE INDEX IF NOT EXISTS idx_tensors_dtype ON tensors(dtype);
"""

META_TYPE_MAP = {
    GGUFValueType.UINT8: "uint8",
    GGUFValueType.INT8: "int8",
    GGUFValueType.UINT16: "uint16",
    GGUFValueType.INT16: "int16",
    GGUFValueType.UINT32: "uint32",
    GGUFValueType.INT32: "int32",
    GGUFValueType.FLOAT32: "float32",
    GGUFValueType.BOOL: "bool",
    GGUFValueType.STRING: "string",
    GGUFValueType.ARRAY: "array",
    GGUFValueType.UINT64: "uint64",
    GGUFValueType.INT64: "int64",
    GGUFValueType.FLOAT64: "float64",
}


def _extract_field_value(field):
    """Extract a Python value from a GGUFReader field."""
    if not field.types:
        return None

    main_type = field.types[0]

    if main_type == GGUFValueType.STRING:
        parts_data = field.parts[field.data[0]]
        return parts_data.tobytes().decode("utf-8", errors="replace")

    if main_type == GGUFValueType.ARRAY:
        if len(field.types) < 2:
            return "[]"
        elem_type = field.types[1]
        values = []
        for idx in field.data:
            part = field.parts[idx]
            if elem_type == GGUFValueType.STRING:
                values.append(part.tobytes().decode("utf-8", errors="replace"))
            else:
                values.append(part.tolist() if hasattr(part, "tolist") else part)
        return str(values)

    part = field.parts[field.data[0]]
    if hasattr(part, "tolist"):
        val = part.tolist()
        return val[0] if isinstance(val, list) and len(val) == 1 else val
    return part


def gguf_to_sqlite(gguf_path: str, db_path: str | None = None, verbose: bool = True) -> str:
    """Convert a GGUF file to a SQLite database.

    Args:
        gguf_path: Path to the input GGUF file.
        db_path: Path for the output SQLite database. Defaults to same name with .db extension.
        verbose: Print progress information.

    Returns:
        Path to the created SQLite database.
    """
    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    if db_path is None:
        db_path = str(gguf_path.with_suffix(".db"))
    db_path = str(db_path)

    if verbose:
        print(f"Reading {gguf_path} ...")

    t0 = time.time()
    reader = GGUFReader(str(gguf_path), mode="r")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.executescript(SCHEMA)

    # Store metadata
    meta_count = 0
    for key, field in reader.fields.items():
        main_type = field.types[0] if field.types else GGUFValueType.STRING
        type_name = META_TYPE_MAP.get(main_type, "unknown")
        value = _extract_field_value(field)
        conn.execute(
            "INSERT OR REPLACE INTO model_meta (key, value, type) VALUES (?, ?, ?)",
            (key, str(value), type_name),
        )
        meta_count += 1

    # Record which dtype system the dtype_id column belongs to, so a later
    # export never mistakes a GGML enum value for a safetensors one.
    for key, value in (
        ("reminis.source_format", "gguf"),
        ("reminis.dtype_system", DTYPE_SYSTEM_GGUF),
    ):
        conn.execute(
            "INSERT OR REPLACE INTO model_meta (key, value, type) VALUES (?, ?, 'string')",
            (key, value),
        )

    if verbose:
        print(f"  Stored {meta_count} metadata fields")

    # Store tensors
    total_bytes = 0
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        shape = [int(x) for x in tensor.shape]
        dtype_enum = tensor.tensor_type
        dtype_name = dtype_enum.name
        n_elements = int(tensor.n_elements)
        n_bytes = int(tensor.n_bytes)
        raw_data = tensor.data.tobytes()

        conn.execute(
            "INSERT OR REPLACE INTO tensors (name, shape, dtype, dtype_id, n_elements, n_bytes, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, json.dumps(shape), dtype_name, dtype_enum.value, n_elements, n_bytes, raw_data),
        )
        total_bytes += n_bytes

        if verbose:
            size_kb = n_bytes / 1024
            unit = "KB" if size_kb < 1024 else "MB"
            size_display = size_kb if unit == "KB" else size_kb / 1024
            print(f"  [{i+1}/{len(reader.tensors)}] {name:50s} {str(shape):20s} {dtype_name:8s} {size_display:8.1f} {unit}")

    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    total_mb = total_bytes / (1024 * 1024)

    if verbose:
        print(f"\nDone. {len(reader.tensors)} tensors ({total_mb:.1f} MB) stored in {db_path}")
        print(f"Time: {elapsed:.1f}s ({total_mb / elapsed:.1f} MB/s)")

    return db_path


def sqlite_to_gguf(db_path: str, gguf_path: str | None = None, verbose: bool = True) -> str:
    """Convert a SQLite database back to a GGUF file.

    Args:
        db_path: Path to the input SQLite database.
        gguf_path: Path for the output GGUF file. Defaults to same name with .gguf extension.
        verbose: Print progress information.

    Returns:
        Path to the created GGUF file.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    if gguf_path is None:
        gguf_path = str(db_path.with_suffix(".gguf"))
    gguf_path = str(gguf_path)

    if verbose:
        print(f"Reading {db_path} ...")

    t0 = time.time()
    conn = sqlite3.connect(str(db_path))

    # dtype_id holds a GGML enum value for GGUF-sourced models and a
    # reminis-local id for safetensors ones. Exporting the latter as GGUF would
    # reinterpret those ids as quantization types and write a corrupt file.
    system = dtype_system(conn)
    if system == DTYPE_SYSTEM_SAFETENSORS:
        conn.close()
        raise ValueError(
            f"{db_path} came from safetensors, whose dtypes are not GGML types. "
            "Export it with `reminis export --format safetensors`, or convert "
            "the original through the llama.cpp toolchain to get a GGUF."
        )

    # Read architecture from metadata
    row = conn.execute("SELECT value FROM model_meta WHERE key = 'general.architecture'").fetchone()
    arch = row[0] if row else "llama"

    writer = GGUFWriter(gguf_path, arch)

    # Write metadata
    meta_rows = conn.execute("SELECT key, value, type FROM model_meta").fetchall()
    meta_count = 0
    for key, value, type_name in meta_rows:
        # GGUF.* are reader-internal; reminis.* are our own bookkeeping and
        # must not leak into the exported file, or a round-trip would grow
        # metadata keys the original never had.
        if key.startswith(("GGUF.", "reminis.")):
            continue
        try:
            _write_meta_value(writer, key, value, type_name)
            meta_count += 1
        except Exception:
            pass

    if verbose:
        print(f"  Wrote {meta_count} metadata fields")

    # Write tensors
    tensor_rows = conn.execute(
        "SELECT name, shape, dtype, dtype_id, n_elements, n_bytes, data FROM tensors ORDER BY id"
    ).fetchall()

    NATIVE_DTYPES = {
        GGMLQuantizationType.F32: np.float32,
        GGMLQuantizationType.F16: np.float16,
    }

    total_bytes = 0
    for i, (name, shape_str, dtype_name, dtype_id, n_elements, n_bytes, data) in enumerate(tensor_rows):
        shape = json.loads(shape_str)
        quant_type = GGMLQuantizationType(dtype_id)

        np_dtype = NATIVE_DTYPES.get(quant_type)
        if np_dtype is not None:
            tensor_data = np.frombuffer(data, dtype=np_dtype).reshape(shape[::-1])
            writer.add_tensor(name, tensor_data)
        else:
            raw_data = np.frombuffer(data, dtype=np.uint8)
            if len(shape) >= 2:
                block_size, type_size = GGML_QUANT_SIZES[quant_type]
                byte_last = (shape[0] // block_size) * type_size
                byte_shape = shape[1:] + [byte_last]
                raw_data = raw_data.reshape(byte_shape)
            writer.add_tensor(name, raw_data, raw_dtype=quant_type)
        total_bytes += n_bytes

        if verbose:
            size_kb = n_bytes / 1024
            unit = "KB" if size_kb < 1024 else "MB"
            size_display = size_kb if unit == "KB" else size_kb / 1024
            print(f"  [{i+1}/{len(tensor_rows)}] {name:50s} {str(shape):20s} {dtype_name:8s} {size_display:8.1f} {unit}")

    conn.close()

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=verbose)
    writer.close()

    elapsed = time.time() - t0
    total_mb = total_bytes / (1024 * 1024)

    if verbose:
        print(f"\nDone. {len(tensor_rows)} tensors ({total_mb:.1f} MB) written to {gguf_path}")
        print(f"Time: {elapsed:.1f}s ({total_mb / elapsed:.1f} MB/s)")

    return gguf_path


def _write_meta_value(writer: GGUFWriter, key: str, value: str, type_name: str):
    """Write a metadata value to the GGUF writer using the appropriate typed method."""
    # Use built-in helper methods for known keys
    helpers = {
        "general.name": ("add_name", str),
        "general.architecture": None,  # already set via constructor
        "general.quantization_version": ("add_quantization_version", int),
    }

    if key in helpers:
        entry = helpers[key]
        if entry is None:
            return
        method_name, cast = entry
        getattr(writer, method_name)(cast(value))
        return

    # Generic fallback by type
    if type_name in ("uint8", "uint16", "uint32", "uint64"):
        writer.add_uint32(key, int(value))
    elif type_name in ("int8", "int16", "int32", "int64"):
        writer.add_int32(key, int(value))
    elif type_name in ("float32", "float64"):
        writer.add_float32(key, float(value))
    elif type_name == "bool":
        writer.add_bool(key, value.lower() in ("true", "1"))
    elif type_name == "string":
        writer.add_string(key, value)
    elif type_name == "array":
        pass  # arrays are complex; skip for now to avoid corruption
