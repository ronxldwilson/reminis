"""Convert between model files and SQLite databases.

GGUF lives here; safetensors lives in ``safetensors_io`` and shares this
schema. ``shape`` is stored reversed relative to the data layout for every
format, because that is GGUF's convention and the diff, low-rank, and viewer
code all read it that way.
"""

import ast
import json
import mmap
import sqlite3
import struct
import time
from pathlib import Path

import numpy as np

from gguf.gguf_reader import GGUFReader
from gguf import GGUFWriter
from gguf.constants import (
    GGML_QUANT_SIZES,
    GGMLQuantizationType,
    GGUFValueType,
)

from reminis.dtypes import DTYPE_SYSTEM_GGUF, DTYPE_SYSTEM_SAFETENSORS, dtype_system

# Writing a model is one bulk load into a file that does not exist yet, so
# the usual durability machinery protects nothing: there is no earlier state
# of this database to lose, and a crash mid-convert leaves a file you delete
# and rebuild. Turning the journal off and letting the OS schedule the writes
# took the write phase of a 2.5 GB model from 13.9s to 0.8s.
#
# The page size is the one setting that outlives this connection -- it is
# stamped into the file header -- so it was chosen by measuring reads as well
# as writes. On a 2.5 GB model, 64 KB pages beat SQLite's 4 KB default on
# every axis: 5.5x the write throughput, 3.6x sequential read, and 2.8x on
# the small random byte-range reads the expert index does.
BULK_WRITE_PRAGMAS = (
    "PRAGMA page_size=65536",
    "PRAGMA journal_mode=OFF",
    "PRAGMA synchronous=OFF",
    "PRAGMA cache_size=-256000",
)


def open_for_bulk_write(db_path: str) -> sqlite3.Connection:
    """A connection tuned for writing a whole model into a fresh file.

    ``isolation_level=None`` hands transaction control to us, so the load
    runs inside one explicit BEGIN/COMMIT rather than sqlite3's implicit
    per-statement one.
    """
    conn = sqlite3.connect(db_path, isolation_level=None)
    for pragma in BULK_WRITE_PRAGMAS:
        conn.execute(pragma)
    return conn


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


# struct format, width, and numpy dtype for every scalar GGUF value type.
_GGUF_SCALARS = {
    GGUFValueType.UINT8:   ("<B", 1, np.uint8),
    GGUFValueType.INT8:    ("<b", 1, np.int8),
    GGUFValueType.UINT16:  ("<H", 2, np.uint16),
    GGUFValueType.INT16:   ("<h", 2, np.int16),
    GGUFValueType.UINT32:  ("<I", 4, np.uint32),
    GGUFValueType.INT32:   ("<i", 4, np.int32),
    GGUFValueType.FLOAT32: ("<f", 4, np.float32),
    GGUFValueType.BOOL:    ("<?", 1, np.bool_),
    GGUFValueType.UINT64:  ("<Q", 8, np.uint64),
    GGUFValueType.INT64:   ("<q", 8, np.int64),
    GGUFValueType.FLOAT64: ("<d", 8, np.float64),
}

# Every value type by the name the `type` column uses, so an export can map
# a stored name back to the type it was written as.
META_NAME_TO_TYPE = {name: gtype for gtype, name in META_TYPE_MAP.items()}


def _meta_type_name(gtype: GGUFValueType, elem: GGUFValueType | None) -> str:
    """The `type` column's value for one metadata field.

    An array records what it holds -- ``array:string``, ``array:int32`` --
    because the rendered value cannot say. '[[1], [1]]' is the same text
    whether those were int32 or uint32, and an export writing the field back
    has to pick one. Databases written before this carry a bare 'array' and
    are handled by inferring the element type from the values, which is
    right for strings and floats and picks int32 for whole numbers.
    """
    name = META_TYPE_MAP.get(gtype, "unknown")
    if gtype == GGUFValueType.ARRAY and elem is not None:
        return f"{name}:{META_TYPE_MAP.get(elem, 'unknown')}"
    return name


def _meta_array_values(text: str) -> list:
    """The Python values behind a stored array field.

    Both stored shapes parse: a string array is ``['a', 'b']`` and a numeric
    one is ``[[1], [1]]``, each element wrapped because the reference reader
    hands back one-element numpy arrays. repr and literal_eval are an exact
    round trip for str, int and float, so nothing is lost going back.
    """
    values = ast.literal_eval(text)
    if not isinstance(values, list):
        raise ValueError("stored array field is not a list")
    if values and all(isinstance(v, list) and len(v) == 1 for v in values):
        return [v[0] for v in values]
    return values


GGUF_MAGIC_LE = 0x46554747  # 'GGUF'
GGUF_SUPPORTED_VERSIONS = (2, 3)
GGUF_DEFAULT_ALIGNMENT = 32


def _gguf_value(buf, offs: int, gtype: GGUFValueType):
    """One metadata value, already rendered the way the database stores it.

    Returns (text, offset just past the value, element type). The text is
    what ``str(_extract_field_value(field))`` produces for the same field,
    because that is the string every database written before this parser
    existed holds and a round-trip has to keep reproducing.

    The element type is None unless the field is an array. It is returned
    because nothing else records it: the rendered text says an array held
    integers but not whether they were int32 or uint32, and an export that
    has to write the field back needs to know.
    """
    if gtype == GGUFValueType.STRING:
        (n,) = struct.unpack_from("<Q", buf, offs)
        offs += 8
        return bytes(buf[offs:offs + n]).decode("utf-8", errors="replace"), offs + n, None

    scalar = _GGUF_SCALARS.get(gtype)
    if scalar is not None:
        fmt, width, _ = scalar
        (value,) = struct.unpack_from(fmt, buf, offs)
        return str(value), offs + width, None

    if gtype != GGUFValueType.ARRAY:
        raise ValueError(f"Unknown/unhandled field type {gtype}")

    raw_elem, count = struct.unpack_from("<IQ", buf, offs)
    offs += 12
    elem = GGUFValueType(raw_elem)
    # The reference reader reports an empty array as a bare '[]' of no
    # element type, because it derives the type from the first element and
    # there is none. The header does say, but reporting it here would make
    # the two parsers disagree for no gain: an empty array is skipped on
    # export, since the writer cannot encode one.
    if count == 0:
        return "[]", offs, None

    if elem == GGUFValueType.STRING:
        values = []
        append = values.append
        unpack_from = struct.unpack_from
        for _ in range(count):
            (n,) = unpack_from("<Q", buf, offs)
            offs += 8
            append(bytes(buf[offs:offs + n]).decode("utf-8", errors="replace"))
            offs += n
        return "[" + ", ".join(map(repr, values)) + "]", offs, elem

    scalar = _GGUF_SCALARS.get(elem)
    if scalar is None:
        # Nested arrays are legal in the format and absent from every model
        # anyone ships. Refusing sends the caller to the reference reader.
        raise ValueError(f"Unhandled array element type {elem}")
    _, width, np_dtype = scalar
    values = np.frombuffer(buf, dtype=np_dtype, count=count, offset=offs).tolist()
    offs += width * count
    # Each element of a numeric array arrives from the reference reader as a
    # one-element numpy array, so its repr is '[v]' and the whole field reads
    # '[[1], [1], ...]'. Odd, and load-bearing: it is what is already stored.
    return "[" + ", ".join(["[" + repr(v) + "]" for v in values]) + "]", offs, elem


def _parse_gguf_header(buf) -> tuple[list, list, int]:
    """Read a GGUF header straight out of its bytes.

    ``GGUFReader`` builds a numpy view per value, which costs two memmap
    slices for every string in the file. A 128k-token vocabulary is 400k
    such slices, and on Llama-3.2-1B that alone was 7.3s of a 20.6s
    conversion. Reading the same bytes with ``struct`` and one
    ``np.frombuffer`` per numeric array does it in a fraction of that.

    Returns (metadata rows, tensor records, offset of the data block).
    Raises on anything it does not recognise, so the caller can fall back to
    the reference reader rather than guess.
    """
    magic, version = struct.unpack_from("<II", buf, 0)
    if magic != GGUF_MAGIC_LE:
        raise ValueError("GGUF magic invalid")
    if version not in GGUF_SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported GGUF version {version}")

    tensor_count, kv_count = struct.unpack_from("<QQ", buf, 8)
    offs = 24

    meta = [
        ("GGUF.version", str(version), "uint32"),
        ("GGUF.tensor_count", str(tensor_count), "uint64"),
        ("GGUF.kv_count", str(kv_count), "uint64"),
    ]
    alignment = GGUF_DEFAULT_ALIGNMENT
    seen = {key for key, _, _ in meta}

    for _ in range(kv_count):
        (klen,) = struct.unpack_from("<Q", buf, offs)
        offs += 8
        key = bytes(buf[offs:offs + klen]).decode("utf-8")
        offs += klen
        (raw_type,) = struct.unpack_from("<I", buf, offs)
        offs += 4
        gtype = GGUFValueType(raw_type)
        text, offs, elem = _gguf_value(buf, offs, gtype)
        if key in seen:
            raise KeyError(f"Duplicate {key} already in list")
        seen.add(key)
        if key == "general.alignment":
            if gtype != GGUFValueType.UINT32:
                raise ValueError("Bad type for general.alignment field")
            alignment = int(text)
            if alignment == 0 or (alignment & (alignment - 1)) != 0:
                raise ValueError("Invalid alignment: must be a non-zero power of two")
        meta.append((key, text, _meta_type_name(gtype, elem)))

    tensors = []
    names = set()
    for _ in range(tensor_count):
        (nlen,) = struct.unpack_from("<Q", buf, offs)
        offs += 8
        name = bytes(buf[offs:offs + nlen]).decode("utf-8")
        offs += nlen
        (n_dims,) = struct.unpack_from("<I", buf, offs)
        offs += 4
        dims = list(struct.unpack_from(f"<{n_dims}Q", buf, offs))
        offs += 8 * n_dims
        (raw_dtype,) = struct.unpack_from("<I", buf, offs)
        offs += 4
        (rel_offset,) = struct.unpack_from("<Q", buf, offs)
        offs += 8

        if name in names:
            raise ValueError(f"Found duplicated tensor with name {name}")
        names.add(name)

        quant = GGMLQuantizationType(raw_dtype)
        block_size, type_size = GGML_QUANT_SIZES[quant]
        n_elements = 1
        for d in dims:
            n_elements *= int(d)
        n_bytes = n_elements * type_size // block_size
        tensors.append({
            "name": name,
            "shape": [int(d) for d in dims],
            "dtype": quant.name,
            "dtype_id": int(quant),
            "n_elements": n_elements,
            "n_bytes": n_bytes,
            "rel_offset": int(rel_offset),
        })

    padding = offs % alignment
    if padding:
        offs += alignment - padding
    return meta, tensors, offs


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

    # A fresh file, so the bulk-write pragmas apply and no tensor of a
    # previous model can survive into this one.
    Path(db_path).unlink(missing_ok=True)

    with open(gguf_path, "rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            try:
                meta_rows, tensors, data_offset = _parse_gguf_header(buf)
            except Exception as exc:
                # Anything the fast parser does not recognise goes to the
                # reference reader rather than being guessed at.
                if verbose:
                    print(f"  (reading with the reference GGUF reader: {exc})")
                return _gguf_to_sqlite_via_reader(gguf_path, db_path, verbose, t0)

            meta_rows = list(meta_rows) + [
                # Which dtype system the dtype_id column belongs to, so a
                # later export never mistakes a GGML enum value for a
                # safetensors one.
                ("reminis.source_format", "gguf", "string"),
                ("reminis.dtype_system", DTYPE_SYSTEM_GGUF, "string"),
            ]

            conn = open_for_bulk_write(db_path)
            try:
                conn.executescript(SCHEMA)
                conn.execute("BEGIN")
                conn.executemany(
                    "INSERT OR REPLACE INTO model_meta (key, value, type) VALUES (?, ?, ?)",
                    meta_rows,
                )
                if verbose:
                    print(f"  Stored {len(meta_rows)} metadata fields")

                total_bytes = sum(t["n_bytes"] for t in tensors)
                view = memoryview(buf)

                def rows():
                    # A memoryview onto the mapping, so a tensor's bytes go
                    # from the file's pages into the database page without a
                    # full copy into a Python bytes object on the way.
                    #
                    # Each slice is an export of the mapping, and a mapping
                    # with exports outstanding refuses to close. By the time
                    # the generator is resumed sqlite has finished with the
                    # previous row, so that is the moment to hand it back.
                    previous = None
                    try:
                        for i, t in enumerate(tensors):
                            if previous is not None:
                                previous.release()
                                previous = None
                            start = data_offset + t["rel_offset"]
                            blob = view[start:start + t["n_bytes"]]
                            previous = blob
                            if verbose:
                                size_kb = t["n_bytes"] / 1024
                                unit = "KB" if size_kb < 1024 else "MB"
                                shown = size_kb if unit == "KB" else size_kb / 1024
                                print(f"  [{i+1}/{len(tensors)}] {t['name']:50s} "
                                      f"{str(t['shape']):20s} {t['dtype']:8s} "
                                      f"{shown:8.1f} {unit}")
                            yield (t["name"], json.dumps(t["shape"]), t["dtype"],
                                   t["dtype_id"], t["n_elements"], t["n_bytes"], blob)
                    finally:
                        if previous is not None:
                            previous.release()

                try:
                    conn.executemany(
                        "INSERT OR REPLACE INTO tensors (name, shape, dtype, dtype_id, "
                        "n_elements, n_bytes, data) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        rows(),
                    )
                    conn.execute("COMMIT")
                finally:
                    view.release()
            finally:
                conn.close()

    elapsed = time.time() - t0
    total_mb = total_bytes / (1024 * 1024)

    if verbose:
        print(f"\nDone. {len(tensors)} tensors ({total_mb:.1f} MB) stored in {db_path}")
        print(f"Time: {elapsed:.1f}s ({total_mb / elapsed:.1f} MB/s)")

    return db_path


def _gguf_to_sqlite_via_reader(gguf_path, db_path: str, verbose: bool, t0: float) -> str:
    """The original conversion, through the ``gguf`` package's own reader.

    Kept as the fallback for files the fast header parser declines, so a
    format extension it has never seen still converts rather than failing.
    """
    reader = GGUFReader(str(gguf_path), mode="r")

    conn = open_for_bulk_write(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute("BEGIN")

        meta_rows = []
        for key, field in reader.fields.items():
            main_type = field.types[0] if field.types else GGUFValueType.STRING
            elem = field.types[1] if len(field.types) > 1 else None
            meta_rows.append((key, str(_extract_field_value(field)),
                              _meta_type_name(main_type, elem)))
        meta_rows += [
            ("reminis.source_format", "gguf", "string"),
            ("reminis.dtype_system", DTYPE_SYSTEM_GGUF, "string"),
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO model_meta (key, value, type) VALUES (?, ?, ?)",
            meta_rows,
        )
        if verbose:
            print(f"  Stored {len(meta_rows)} metadata fields")

        total_bytes = 0
        for tensor in reader.tensors:
            total_bytes += int(tensor.n_bytes)

        def rows():
            for i, tensor in enumerate(reader.tensors):
                shape = [int(x) for x in tensor.shape]
                n_bytes = int(tensor.n_bytes)
                if verbose:
                    size_kb = n_bytes / 1024
                    unit = "KB" if size_kb < 1024 else "MB"
                    shown = size_kb if unit == "KB" else size_kb / 1024
                    print(f"  [{i+1}/{len(reader.tensors)}] {tensor.name:50s} "
                          f"{str(shape):20s} {tensor.tensor_type.name:8s} "
                          f"{shown:8.1f} {unit}")
                yield (tensor.name, json.dumps(shape), tensor.tensor_type.name,
                       tensor.tensor_type.value, int(tensor.n_elements), n_bytes,
                       tensor.data.tobytes())

        conn.executemany(
            "INSERT OR REPLACE INTO tensors (name, shape, dtype, dtype_id, "
            "n_elements, n_bytes, data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows(),
        )
        conn.execute("COMMIT")
    finally:
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

    # Describe every tensor without reading one. add_tensor() would hold the
    # weights in the writer until the end -- 12 GB of them on a 20B model --
    # and then write each through numpy's tofile, which measured 176 MB/s
    # against a disk that reads at 2.8 GB/s. add_tensor_info() records the
    # same header entry and leaves the bytes to us.
    NATIVE_DTYPES = {
        GGMLQuantizationType.F32: np.float32,
        GGMLQuantizationType.F16: np.float16,
    }

    plan = []
    total_bytes = 0
    for name, shape_str, dtype_name, dtype_id, n_elements, n_bytes in conn.execute(
        "SELECT name, shape, dtype, dtype_id, n_elements, n_bytes FROM tensors ORDER BY id"
    ):
        shape = json.loads(shape_str)
        quant_type = GGMLQuantizationType(dtype_id)
        np_dtype = NATIVE_DTYPES.get(quant_type)

        if np_dtype is not None:
            # Shapes go to the writer in numpy order; it reverses them back.
            writer.add_tensor_info(name, shape[::-1], np.dtype(np_dtype),
                                   n_bytes, raw_dtype=quant_type)
        else:
            byte_shape = list(shape[::-1])
            if len(shape) >= 2:
                block_size, type_size = GGML_QUANT_SIZES[quant_type]
                byte_shape = shape[1:] + [(shape[0] // block_size) * type_size]
            writer.add_tensor_info(name, byte_shape, np.dtype(np.uint8),
                                   n_bytes, raw_dtype=quant_type)

        plan.append((name, shape, dtype_name, n_bytes))
        total_bytes += n_bytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # write_ti_data_to_file laid the offsets out as a running total of
    # ggml_pad(nbytes), starting from the first aligned byte after the
    # header. Writing the data has to follow exactly that, so the padding
    # here is not cosmetic -- it is what the offsets already promised.
    if writer.fout is None or len(writer.fout) != 1:
        conn.close()
        writer.close()
        raise ValueError(
            "This writer split the output across shards, which this export "
            "path does not handle."
        )
    fout = writer.fout[0]
    writer.write_padding(fout, fout.tell())

    for i, (name, shape, dtype_name, n_bytes) in enumerate(plan):
        (blob,) = conn.execute(
            "SELECT data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        fout.write(blob)
        writer.write_padding(fout, n_bytes)
        del blob  # one tensor resident at a time, not the whole model

        if verbose:
            size_kb = n_bytes / 1024
            unit = "KB" if size_kb < 1024 else "MB"
            size_display = size_kb if unit == "KB" else size_kb / 1024
            print(f"  [{i+1}/{len(plan)}] {name:50s} {str(shape):20s} {dtype_name:8s} {size_display:8.1f} {unit}")

    fout.flush()
    conn.close()
    writer.close()

    elapsed = time.time() - t0
    total_mb = total_bytes / (1024 * 1024)

    if verbose:
        print(f"\nDone. {len(plan)} tensors ({total_mb:.1f} MB) written to {gguf_path}")
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
    elif type_name == "array" or type_name.startswith("array:"):
        # Skipping these was silently dropping the tokenizer. Its vocabulary
        # and merges are string arrays, so an exported file had correct
        # weights and nothing to tokenize with, and llama.cpp refused it with
        # "cannot find tokenizer merges in model file".
        values = _meta_array_values(value)
        if not values:
            # The writer cannot encode an empty array -- it derives the
            # element type from the first element. Nothing ships one.
            return
        sub_type = None
        if ":" in type_name:
            sub_type = META_NAME_TO_TYPE.get(type_name.split(":", 1)[1])
        if sub_type is None:
            # A database written before the element type was recorded. The
            # values still say what they are, except that every whole number
            # reads as int32.
            sub_type = GGUFValueType.get_type(values[0])
        writer.add_key_value(key, values, GGUFValueType.ARRAY, sub_type=sub_type)
