"""Compare two model databases and produce reusable delta packs.

A delta pack stores only what changed between a base model and a target model.
Applying a pack to the base reconstructs the target, so a fine-tune can be
distributed as a small migration instead of a full model copy.
"""

import hashlib
import json
import sqlite3
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import zstandard

from gguf.constants import GGMLQuantizationType

from reminis.dtypes import from_float32, is_float_dtype, to_float32

# zstd level 1 measured at 547 MB/s on XOR delta data, versus 4 MB/s for zlib
# level 6 -- and it produces a *smaller* payload (32.1 MB vs 32.5 MB on a 54 MB
# tensor). Delta data is high-entropy XORed float mantissas, so no level buys
# much size; spending time on it is pure loss. Higher zstd levels are worse
# trades still: level 19 took 60s to save 4%.
ZSTD_LEVEL = 1

_compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
_decompressor = zstandard.ZstdDecompressor()
_local = threading.local()


def _compress(data: bytes) -> bytes:
    return _compressor.compress(data)


def _get_decompressor() -> zstandard.ZstdDecompressor:
    """A per-thread decompressor, since ZstdDecompressor is not thread-safe."""
    d = getattr(_local, "decompressor", None)
    if d is None:
        d = zstandard.ZstdDecompressor()
        _local.decompressor = d
    return d


def _decompress(data: bytes, encoding: str) -> bytes:
    """Decompress a payload, honouring the codec named in its encoding.

    Packs written before 0.3.0 use zlib; those encodings are still read so
    existing packs keep working. ZstdCompressor.compress() writes the content
    size into the frame header, so no size hint is needed here.
    """
    if encoding.endswith("_zlib"):
        return zlib.decompress(data)
    return _get_decompressor().decompress(data)

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
    rank          INTEGER,
    rel_error     REAL,
    data          BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_deltas_name ON deltas(tensor_name);
"""

# Low-rank factors are stored float16 by default. Measured against float32 on
# LoRA-shaped deltas: half the size for indistinguishable error (1.15e-04 vs
# 1.13e-04). Folding sqrt(S) into both factors keeps their dynamic range
# balanced, and the reconstruction is rounded back to the tensor's own dtype
# anyway, so the factor rounding disappears into that final step.
#
# float32 is still used when the factors would not survive float16 -- see
# _pick_factor_dtype. The chosen dtype is recorded per tensor in the payload
# header, so decoding never has to guess.
LOWRANK_FACTOR_DTYPES = {"f16": np.float16, "f32": np.float32}
_F16_MAX = float(np.finfo(np.float16).max)


def _pick_factor_dtype(*factors: np.ndarray) -> str:
    """Use float16 for the factors unless doing so would lose or break values.

    Overflow to inf is the failure that matters: float16 tops out around
    65504, and a factor above that would decode to garbage rather than to a
    slightly wrong number.
    """
    for f in factors:
        if not np.all(np.isfinite(f)):
            raise ValueError("low-rank factors contain non-finite values")
        if np.max(np.abs(f)) > _F16_MAX:
            return "f32"
    return "f16"


def _randomized_svd(a: np.ndarray, rank: int, n_iter: int = 4, seed: int = 0):
    """Truncated SVD via random projection.

    A full np.linalg.svd is O(min(m,n)^2 * max(m,n)) and becomes the dominant
    cost on large tensors, where we only ever want the leading handful of
    components. The seed is fixed so packs are reproducible.
    """
    rng = np.random.default_rng(seed)
    m, n = a.shape
    p = min(rank + 10, min(m, n))
    # numpy's SIMD matmul path raises spurious divide/overflow/invalid warnings
    # on some builds even when every output is finite. Callers verify the
    # result with np.isfinite rather than relying on these.
    with np.errstate(all="ignore"):
        y = a @ rng.standard_normal((n, p)).astype(np.float32)
        for _ in range(n_iter):
            y, _ = np.linalg.qr(y)
            y = a @ (a.T @ y)
        q, _ = np.linalg.qr(y)
        ub, s, vt = np.linalg.svd(q.T @ a, full_matrices=False)
        return (q @ ub)[:, :rank], s[:rank], vt[:rank]


def _encode_lowrank(
    a_blob: bytes,
    b_blob: bytes,
    dtype: str,
    shape: list,
    tolerance: float,
    budget_bytes: int,
):
    """Try to express the delta as low-rank factors within an error tolerance.

    Returns (payload, rank, rel_error) or None when low-rank is not applicable
    or not worth it. `budget_bytes` is the size to beat -- normally whatever
    the lossless encoding achieved, so we never grow a pack to make it lossy.

    Only 2D float tensors qualify: SVD needs a matrix, and quantized bytes
    cannot be decoded to values to decompose in the first place.
    """
    if not is_float_dtype(dtype) or len(shape) != 2:
        return None

    # Shape is stored reversed relative to the data layout (GGUF's convention,
    # which reminis keeps for every format).
    m, n = shape[1], shape[0]
    a = to_float32(a_blob, dtype).reshape(m, n)
    b = to_float32(b_blob, dtype).reshape(m, n)
    delta = b - a

    delta_norm = float(np.linalg.norm(delta))
    if delta_norm == 0:
        return None

    # Above this rank the factors cost more than the lossless payload, so
    # there is no point decomposing further. Sized against float16 factors,
    # the common case.
    max_rank = budget_bytes // ((m + n) * 2)
    max_rank = int(min(max_rank, min(m, n) - 1))
    if max_rank < 1:
        return None

    u, s, vt = _randomized_svd(delta, max_rank)

    # Smallest rank meeting the tolerance. Truncation error at rank r is the
    # norm of the discarded singular values.
    tail = np.sqrt(np.maximum(np.cumsum((s**2)[::-1])[::-1], 0.0))
    rel = tail / delta_norm
    within = np.nonzero(rel <= tolerance)[0]
    if len(within) == 0:
        return None
    rank = int(within[0]) + 1

    # Fold the singular values in so apply is a plain matmul.
    sqrt_s = np.sqrt(s[:rank])
    left_f32 = u[:, :rank] * sqrt_s
    right_f32 = sqrt_s[:, None] * vt[:rank]

    try:
        factor_key = _pick_factor_dtype(left_f32, right_f32)
    except ValueError:
        return None  # degenerate decomposition; keep the lossless encoding

    factor_dtype = LOWRANK_FACTOR_DTYPES[factor_key]
    left = left_f32.astype(factor_dtype)
    right = right_f32.astype(factor_dtype)

    # Measure the error actually achieved, after factor rounding, rather than
    # trusting the spectral estimate.
    with np.errstate(all="ignore"):
        approx = a + (left.astype(np.float32) @ right.astype(np.float32))
    if not np.all(np.isfinite(approx)):
        return None
    # Round-trip through the tensor's own dtype, so the error measured is the
    # error apply will actually deliver.
    rebuilt = to_float32(from_float32(approx, dtype), dtype).reshape(m, n)
    if not np.all(np.isfinite(rebuilt)):
        return None
    achieved = float(np.linalg.norm(rebuilt - b) / np.linalg.norm(b))
    if achieved > tolerance:
        return None  # rounding pushed it out of budget

    header = json.dumps({"rank": rank, "m": m, "n": n, "fdt": factor_key}).encode()
    payload = _compress(
        len(header).to_bytes(4, "little") + header + left.tobytes() + right.tobytes()
    )
    if len(payload) >= budget_bytes:
        return None

    return payload, rank, achieved


def _decode_lowrank(raw: bytes, base_blob: bytes, dtype: str) -> bytes:
    """Rebuild a tensor from base weights plus low-rank delta factors."""
    header_len = int.from_bytes(raw[:4], "little")
    meta = json.loads(raw[4 : 4 + header_len])
    rank, m, n = meta["rank"], meta["m"], meta["n"]
    factor_dtype = LOWRANK_FACTOR_DTYPES[meta.get("fdt", "f32")]

    body = np.frombuffer(raw, dtype=factor_dtype, offset=4 + header_len)
    left = body[: m * rank].reshape(m, rank).astype(np.float32)
    right = body[m * rank : m * rank + rank * n].reshape(rank, n).astype(np.float32)

    base = to_float32(base_blob, dtype).reshape(m, n)
    with np.errstate(all="ignore"):
        rebuilt = base + (left @ right)

    # A non-finite value here means the pack is corrupt. Failing loudly beats
    # writing NaNs into a model that will only misbehave much later.
    if not np.all(np.isfinite(rebuilt)):
        raise ValueError(
            "Low-rank reconstruction produced non-finite values; the delta pack is corrupt."
        )
    return from_float32(rebuilt, dtype)


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


class _BackgroundHash:
    """A sha256 fed from another thread, so hashing overlaps producing the bytes.

    hashlib drops the interpreter lock for updates of any size worth
    threading, so a caller that is busy reading and comparing blobs can hand
    them here and keep going.

    The queue is bounded on purpose: the things being hashed are whole
    tensors, and an unbounded one would happily hold the entire model in
    memory while the hasher fell behind.
    """

    __slots__ = ("_queue", "_digest", "_thread")

    def __init__(self, depth: int = 4):
        from queue import Queue

        self._queue = Queue(maxsize=depth)
        self._digest = hashlib.sha256()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            chunk = self._queue.get()
            if chunk is None:
                return
            self._digest.update(chunk)

    def update(self, data):
        self._queue.put(data)

    def hexdigest(self) -> str:
        self._queue.put(None)
        self._thread.join()
        return self._digest.hexdigest()


def _weights_hash_threaded(db_path: str) -> str:
    """Hash weight data with I/O on a background thread.

    SHA-256 and SQLite blob reads both release the GIL, so a reader thread
    keeps blobs ready while the main thread hashes the previous one.
    """
    from queue import Queue
    from threading import Thread

    q = Queue(maxsize=8)

    def _reader():
        c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        for (blob,) in c.execute("SELECT data FROM tensors ORDER BY name"):
            q.put(blob)
        c.close()
        q.put(None)

    t = Thread(target=_reader, daemon=True)
    t.start()
    h = hashlib.sha256()
    while True:
        blob = q.get()
        if blob is None:
            break
        h.update(blob)
    t.join()
    return h.hexdigest()


def _tensor_delta_stats(a_blob: bytes, b_blob: bytes, dtype: str):
    """Compute change statistics between two tensors of the same dtype.

    Returns (stats_dict, delta_array_or_None). The delta array is None for
    quantized types, where raw bytes cannot be subtracted meaningfully.
    """
    if a_blob == b_blob:
        return {"identical": True}, None

    if not is_float_dtype(dtype):
        # Quantized or integer: we can tell that it changed, but not by how much.
        return {
            "identical": False,
            "numeric": False,
            "n_changed": None,
            "pct_changed": None,
            "l2_norm": None,
            "max_abs_delta": None,
            "mean_delta": None,
        }, None

    a = to_float32(a_blob, dtype)
    b = to_float32(b_blob, dtype)

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


# Float dtypes whose delta is worth splitting into byte planes, and the width
# of one value in bytes. Only 2-byte floats are handled: their exponent lives
# almost entirely in one byte and their noisy low mantissa bits in the other,
# which is what makes the split pay.
_PLANE_WIDTH = {"F16": 2, "BF16": 2}


def _split_planes(data: bytes, width: int) -> list[bytes]:
    """Deinterleave a byte string into `width` streams, one per byte position.

    Indexing by byte offset rather than by shifting a decoded integer keeps
    this independent of the host's endianness: the streams are defined by the
    stored layout, which is little-endian in both GGUF and safetensors.
    """
    arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, width)
    return [arr[:, i].tobytes() for i in range(width)]


def _merge_planes(planes: list[bytes], width: int) -> bytes:
    out = np.empty((len(planes[0]), width), dtype=np.uint8)
    for i, plane in enumerate(planes):
        out[:, i] = np.frombuffer(plane, dtype=np.uint8)
    return out.tobytes()


def _frame(parts: list[bytes]) -> bytes:
    """Length-prefix each compressed stream so decode needs no side channel."""
    out = bytearray()
    for part in parts:
        chunk = _compress(part)
        out += len(chunk).to_bytes(4, "little")
        out += chunk
    return bytes(out)


def _unframe(payload: bytes, count: int) -> list[bytes]:
    parts, offset = [], 0
    for _ in range(count):
        if offset + 4 > len(payload):
            raise ValueError("Truncated bit-plane payload")
        size = int.from_bytes(payload[offset:offset + 4], "little")
        offset += 4
        if offset + size > len(payload):
            raise ValueError("Truncated bit-plane payload")
        parts.append(_get_decompressor().decompress(payload[offset:offset + size]))
        offset += size
    return parts


def _encode_bitplane(delta: np.ndarray, dtype: str) -> bytes | None:
    """Compress an XOR delta as separate byte planes rather than as one stream.

    A 16-bit float interleaves a highly predictable exponent with mantissa bits
    that a fine-tune randomises, so a general-purpose compressor sees the
    structure it could exploit chopped up by noise every other byte. Splitting
    the two apart lets the exponent plane form the long runs it deserves while
    the mantissa plane is left to cost what it costs.

    Measured on SmolLM-135M against its own instruct fine-tune -- the case
    where every weight moved -- this takes the pack from 58.3% of the raw
    weights to 50.4%, against an order-0 entropy floor of 46.5%.

    Returns None when the dtype is not one this helps, leaving the caller's
    other candidates to compete.
    """
    width = _PLANE_WIDTH.get(dtype)
    if width is None or delta.nbytes % width:
        return None
    return _frame(_split_planes(delta.tobytes(), width))


def _decode_bitplane(payload: bytes, dtype: str, name: str) -> bytes:
    width = _PLANE_WIDTH.get(dtype)
    if width is None:
        raise ValueError(
            f"Delta pack stores tensor '{name}' as bit planes, but its dtype "
            f"'{dtype}' has no plane layout. The pack is inconsistent."
        )
    planes = _unframe(payload, width)
    if len({len(p) for p in planes}) != 1:
        raise ValueError(f"Bit planes for '{name}' have mismatched lengths")
    return _merge_planes(planes, width)


def _encode_delta(a_blob: bytes, b_blob: bytes, dtype: str) -> tuple[str, bytes]:
    """Pick the smaller of a compressed XOR delta or a compressed replacement.

    XOR is used rather than arithmetic subtraction because it is exactly
    reversible for every dtype. An arithmetic float delta is not: `b - a`
    generally is not representable in the tensor's own dtype, so `a + delta`
    lands a rounding step away from `b`. XOR also works on quantized tensors,
    whose bytes cannot be subtracted meaningfully at all.

    Bytes that did not change XOR to zero, so runs of unchanged data compress
    away and the payload tracks how much actually moved.

    Returns (encoding_name, payload). 'xor_zstd' payloads are XORed against the
    base tensor on apply; 'replace_zstd' payloads overwrite it outright.
    """
    replacement = _compress(b_blob)

    if len(a_blob) != len(b_blob):
        return "replace_zstd", replacement

    a_arr = np.frombuffer(a_blob, dtype=np.uint8)
    b_arr = np.frombuffer(b_blob, dtype=np.uint8)
    delta = np.bitwise_xor(a_arr, b_arr)

    candidates = [("xor_zstd", _compress(delta.tobytes())),
                  ("replace_zstd", replacement)]

    planes = _encode_bitplane(delta, dtype)
    if planes is not None:
        candidates.append(("bitplane_zstd", planes))

    # Ties go to the earliest candidate, which keeps a tensor that gains
    # nothing from the split on the plain encoding older readers understand.
    return min(candidates, key=lambda c: len(c[1]))


def _choose_encoding(
    a_blob: bytes,
    b_blob: bytes,
    dtype: str,
    shape: list,
    lossy_tolerance: float | None,
):
    """Pick the smallest encoding for one tensor.

    Lossless is computed first and always available; low-rank is only tried
    when explicitly enabled, and only wins if it beats the lossless size while
    staying inside the tolerance. A pack never grows in order to become lossy.

    Returns (encoding, payload, rank, rel_error).
    """
    encoding, payload = _encode_delta(a_blob, b_blob, dtype)

    if lossy_tolerance is None:
        return encoding, payload, None, None

    attempt = _encode_lowrank(a_blob, b_blob, dtype, shape, lossy_tolerance, len(payload))
    if attempt is None:
        return encoding, payload, None, None

    lr_payload, rank, rel_error = attempt
    return "lowrank_zstd", lr_payload, rank, rel_error


def diff_models(
    db_a: str,
    db_b: str,
    output_path: str | None = None,
    verbose: bool = True,
    lossy_tolerance: float | None = None,
) -> dict:
    """Compare two model databases, optionally writing a delta pack.

    Args:
        db_a: Path to the base model database.
        db_b: Path to the target model database.
        output_path: If given, write a delta pack that turns A into B.
        verbose: Print a progress report.
        lossy_tolerance: Enable low-rank encoding, allowing at most this
            relative reconstruction error per tensor (e.g. 0.01 for 1%).
            None means lossless only. Low-rank helps when the delta is
            genuinely low-rank, as a LoRA fine-tune's is; on a full fine-tune
            it rarely wins and the lossless path is kept automatically.

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
    lowrank_count = 0
    worst_rel_error = 0.0

    # A pack records the hash of both models' weights. Computing those with
    # _weights_hash means reading each model a second time, after this loop
    # has already held every byte of both -- on a 2.5 GB model that is 5 GB
    # of reading to learn something the loop could have accumulated for
    # free.
    #
    # It is only free when the two models hold the same tensors: the hash is
    # defined over every tensor in name order, and this loop walks the
    # intersection. When one side has a tensor the other lacks, the orders
    # part company and the hashes are computed separately below.
    # Each hash runs on its own thread, so the two overlap each other and
    # the reading and comparing here. Hashing 5 GB is a second or two of
    # pure sha256 whichever way it is arranged; this is what stops that
    # being a second or two nothing else is happening.
    hashing_inline = bool(output_path) and not only_in_a and not only_in_b
    hash_a = _BackgroundHash() if hashing_inline else None
    hash_b = _BackgroundHash() if hashing_inline else None

    for name in shared:
        row_a = conn_a.execute(
            "SELECT shape, dtype, dtype_id, n_elements, data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        row_b = conn_b.execute(
            "SELECT shape, dtype, dtype_id, n_elements, data FROM tensors WHERE name = ?", (name,)
        ).fetchone()

        shape_a, dtype_a, dtype_id_a, n_elements_a, blob_a = row_a
        shape_b, dtype_b, dtype_id_b, n_elements_b, blob_b = row_b

        if hash_a is not None:
            hash_a.update(blob_a)
            hash_b.update(blob_b)

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
            encoding, payload, rank, rel_error = _choose_encoding(
                blob_a, blob_b, dtype_b, json.loads(shape_b), lossy_tolerance
            )
            total_raw_bytes += len(blob_b)
            total_stored_bytes += len(payload)
            if rank is not None:
                lowrank_count += 1
                worst_rel_error = max(worst_rel_error, rel_error)
                entry["rank"] = rank
                entry["rel_error"] = rel_error

            out_conn.execute(
                "INSERT INTO deltas (tensor_name, shape, dtype, dtype_id, n_elements, "
                "n_changed, pct_changed, l2_norm, max_abs_delta, mean_delta, "
                "encoding, raw_bytes, stored_bytes, rank, rel_error, data) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    name, shape_b, dtype_b, dtype_id_b, n_elements_b,
                    stats.get("n_changed"), stats.get("pct_changed"),
                    stats.get("l2_norm"), stats.get("max_abs_delta"), stats.get("mean_delta"),
                    encoding, len(blob_b), len(payload), rank, rel_error, payload,
                ),
            )

    # Tensors present only in the target must ship in full.
    for name in only_in_b:
        row = conn_b.execute(
            "SELECT shape, dtype, dtype_id, n_elements, data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        shape, dtype, dtype_id, n_elements, blob = row
        if out_conn is not None:
            payload = _compress(blob)
            total_raw_bytes += len(blob)
            total_stored_bytes += len(payload)
            out_conn.execute(
                "INSERT INTO deltas (tensor_name, shape, dtype, dtype_id, n_elements, "
                "encoding, raw_bytes, stored_bytes, data) VALUES (?,?,?,?,?,?,?,?,?)",
                (name, shape, dtype, dtype_id, n_elements, "replace_zstd", len(blob), len(payload), payload),
            )

    total_b_bytes = conn_b.execute("SELECT SUM(n_bytes) FROM tensors").fetchone()[0] or 0

    if out_conn is not None:
        if hash_a is not None:
            base_weights_hash = hash_a.hexdigest()
            target_weights_hash = hash_b.hexdigest()
        else:
            # The tensor sets differ, so the loop above could not accumulate
            # these in the right order. They are independent of each other,
            # so read the two models at once rather than one after the other.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_a = pool.submit(_weights_hash_threaded, str(path_a))
                fut_b = pool.submit(_weights_hash_threaded, str(path_b))
                base_weights_hash = fut_a.result()
                target_weights_hash = fut_b.result()

        meta = {
            "base_fingerprint": fp_a,
            "target_fingerprint": fp_b,
            "base_weights_hash": base_weights_hash,
            "target_weights_hash": target_weights_hash,
            "base_file": path_a.name,
            "target_file": path_b.name,
            "tensors_changed": str(len(changed)),
            "tensors_identical": str(identical_count),
            "tensors_removed": json.dumps(only_in_a),
            "lossy": "true" if lowrank_count else "false",
            "lowrank_tensors": str(lowrank_count),
            "max_rel_error": repr(worst_rel_error),
            "reminis_version": _version(),
        }
        out_conn.commit()

        # A lossy pack cannot reproduce the target byte for byte, but its
        # reconstruction is deterministic -- so record the hash of what apply
        # will actually produce. That keeps apply exactly verifiable; the
        # divergence from the true target is carried separately as an error
        # bound rather than being waved through.
        meta["reconstructed_weights_hash"] = (
            _reconstruction_hash(conn_a, out_conn, only_in_a)
            if lowrank_count
            else meta["target_weights_hash"]
        )

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
        "lowrank_tensors": lowrank_count,
        "max_rel_error": worst_rel_error,
        "lossy": lowrank_count > 0,
        "elapsed": time.time() - t0,
    }

    if verbose:
        _print_diff_report(summary, output_path)

    return summary


KNOWN_ENCODINGS = {
    "xor_zstd", "replace_zstd", "lowrank_zstd", "bitplane_zstd",
    "xor_zlib", "replace_zlib",  # written before 0.3.0
}


def _decode_tensor(encoding: str, payload: bytes, base_blob: bytes | None, dtype: str, name: str) -> bytes:
    """Reconstruct one tensor's bytes from its delta payload.

    Shared by apply and by the reconstruction-hash pass so the two can never
    disagree about what a pack produces.
    """
    if encoding not in KNOWN_ENCODINGS:
        raise ValueError(
            f"Delta pack uses unknown encoding '{encoding}' for tensor '{name}'. "
            "It was likely written by a newer version of reminis."
        )

    # Bit planes carry their own framing, so they are unpacked before the
    # single-stream path gets a chance to treat the payload as one frame.
    if encoding == "bitplane_zstd":
        raw = _decode_bitplane(payload, dtype, name)
    else:
        raw = _decompress(payload, encoding)

    if encoding.startswith("replace_"):
        return raw

    if base_blob is None:
        raise ValueError(f"Delta references tensor '{name}' missing from the base model")

    if encoding == "lowrank_zstd":
        return _decode_lowrank(raw, base_blob, dtype)

    base_arr = np.frombuffer(base_blob, dtype=np.uint8)
    delta_arr = np.frombuffer(raw, dtype=np.uint8)
    if len(base_arr) != len(delta_arr):
        raise ValueError(
            f"Size mismatch applying delta to '{name}': "
            f"base is {len(base_arr)} bytes, delta expects {len(delta_arr)}"
        )
    return np.bitwise_xor(base_arr, delta_arr).tobytes()


def _reconstruction_hash(base_conn, pack_conn, removed: list) -> str:
    """Hash the model that applying this pack to the base will produce.

    Walks tensors in the same order as _weights_hash, substituting each
    pack entry for its base counterpart, holding one tensor at a time.
    """
    entries = {
        r[0]: (r[1], r[2], r[3])
        for r in pack_conn.execute("SELECT tensor_name, encoding, dtype, data FROM deltas")
    }
    dropped = set(removed)

    names = sorted(
        {r[0] for r in base_conn.execute("SELECT name FROM tensors")} | set(entries)
    )

    h = hashlib.sha256()
    for name in names:
        if name in dropped:
            continue
        row = base_conn.execute("SELECT data FROM tensors WHERE name = ?", (name,)).fetchone()
        base_blob = row[0] if row else None
        if name in entries:
            encoding, dtype, payload = entries[name]
            h.update(_decode_tensor(encoding, payload, base_blob, dtype, name))
        else:
            h.update(base_blob)
    return h.hexdigest()


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
        if s["lowrank_tensors"]:
            print(
                f"    Encoding:    {s['lowrank_tensors']} of {s['changed']} tensors "
                f"low-rank (LOSSY), rest lossless"
            )
            print(f"    Worst error: {s['max_rel_error']:.2e} relative, per tensor")
        else:
            print("    Encoding:    lossless")

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
    import os
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    base_path, delta_path = Path(base_db), Path(delta_db)
    for p in (base_path, delta_path):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    t0 = time.time()
    Path(output_db).unlink(missing_ok=True)

    delta_conn = sqlite3.connect(str(delta_path))
    meta = dict(delta_conn.execute("SELECT key, value FROM delta_meta"))

    # Phase 1: copy the base file and hash it in parallel.
    # Both operations read the same source and both release the GIL, so
    # they overlap almost completely on an SSD.
    if verify:
        expected_base = meta.get("base_weights_hash")
        with ThreadPoolExecutor(max_workers=2) as pool:
            hash_future = pool.submit(_weights_hash_threaded, str(base_path))
            copy_future = pool.submit(shutil.copyfile, base_path, output_db)
            copy_future.result()
            actual = hash_future.result()
        if expected_base and actual != expected_base:
            delta_conn.close()
            Path(output_db).unlink(missing_ok=True)
            raise ValueError(
                "Base model does not match the one this delta pack was built from.\n"
                f"  expected weights hash: {expected_base[:16]}...\n"
                f"  actual weights hash:   {actual[:16]}...\n"
                "Applying it would produce a corrupt model."
            )
        if verbose:
            print(f"Base verified against pack ({meta.get('base_file', 'unknown')})")
    else:
        shutil.copyfile(base_path, output_db)

    # This database is a copy taken moments ago, so a rollback journal
    # protects a state nothing would ever want back: an apply that fails is
    # deleted and rerun, which is exactly what the journal would have
    # restored. Measured writing a 1.3 GB delta, dropping it took the write
    # from 7.05s to 1.79s.
    #
    # Deliberately no cache_size here. The converter raises it because it
    # builds a file from nothing; raising it on this path measured slower
    # (2.22s against 1.79s), so the default is left alone. The page size is
    # inherited from the base file and cannot be set on a populated
    # database, so a base converted by 0.27.0 or later writes faster here
    # for free.
    out_conn = sqlite3.connect(output_db, isolation_level=None)
    out_conn.execute("PRAGMA journal_mode=OFF")
    out_conn.execute("PRAGMA synchronous=OFF")

    # Drop tensors the target no longer has.
    removed = json.loads(meta.get("tensors_removed", "[]"))
    for name in removed:
        out_conn.execute("DELETE FROM tensors WHERE name = ?", (name,))

    rows = delta_conn.execute(
        "SELECT tensor_name, shape, dtype, dtype_id, n_elements, encoding, data FROM deltas"
    ).fetchall()
    delta_conn.close()

    # Phase 2: decode deltas in parallel, write sequentially.
    # Decompression (zstd) and XOR (numpy) both release the GIL, so worker
    # threads get real parallelism. Read base blobs from the original file
    # to avoid contention with the output connection.
    base_conn = sqlite3.connect(f"file:{base_db}?mode=ro", uri=True)
    base_blobs = {}
    for name, _shape, _dtype, _dtype_id, _n_elements, encoding, _payload in rows:
        if encoding.startswith("replace"):
            continue
        existing = base_conn.execute(
            "SELECT data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        base_blobs[name] = existing[0] if existing else None
    base_conn.close()

    def _decode_one(args):
        name, encoding, payload, dtype = args
        return _decode_tensor(encoding, payload, base_blobs.get(name), dtype, name)

    work = [
        (name, encoding, payload, dtype)
        for name, _shape, dtype, _dtype_id, _n_elements, encoding, payload in rows
    ]
    workers = min(os.cpu_count() or 4, len(work), 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        decoded = list(pool.map(_decode_one, work))
    base_blobs.clear()

    out_conn.execute("BEGIN")
    for i, (name, shape, dtype, dtype_id, n_elements, _encoding, _payload) in enumerate(rows):
        new_blob = decoded[i]
        out_conn.execute(
            "INSERT INTO tensors (name, shape, dtype, dtype_id, n_elements, n_bytes, data) "
            "VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET shape=excluded.shape, dtype=excluded.dtype, "
            "dtype_id=excluded.dtype_id, n_elements=excluded.n_elements, "
            "n_bytes=excluded.n_bytes, data=excluded.data",
            (name, shape, dtype, dtype_id, n_elements, len(new_blob), new_blob),
        )
    out_conn.execute("COMMIT")
    del decoded

    is_lossy = meta.get("lossy") == "true"

    # Phase 3: verify the result with pipelined hashing.
    if verify:
        expected = meta.get("reconstructed_weights_hash") or meta.get("target_weights_hash")
        if expected:
            actual = _weights_hash_threaded(output_db)
            if actual != expected:
                out_conn.close()
                raise ValueError(
                    "Result does not match the hash recorded in the pack.\n"
                    f"  expected: {expected[:16]}...\n"
                    f"  actual:   {actual[:16]}..."
                )
            if verbose:
                print(
                    "Result verified against reconstruction hash"
                    if is_lossy
                    else "Result verified against target hash"
                )

    out_conn.close()

    if verbose:
        print(f"Applied {len(rows)} tensor updates -> {output_db}")
        if is_lossy:
            err = float(meta.get("max_rel_error", "0"))
            print(
                f"NOTE: this is a lossy pack ({meta.get('lowrank_tensors', '?')} tensors "
                f"low-rank encoded). The result is not byte-identical to the original "
                f"target; worst per-tensor relative error is {err:.2e}."
            )
        print(f"Took {time.time() - t0:.1f}s")

    return output_db
