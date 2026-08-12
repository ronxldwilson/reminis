"""Merge several models into one, aligning their tensors with SQL.

Model merging is normally a bespoke script: load two checkpoints into memory,
match parameter names by hand, average, save. Once the weights are in SQLite
the matching half of that is a join -- ``ATTACH`` every model onto one
connection and ask which tensor names line up, with which shapes, in which
dtypes. That query is the merge plan, and it is written here as SQL rather
than as a Python loop over dictionaries.

The arithmetic itself is still numpy: SQLite has no vector types, so each
aligned row's blob is decoded to float32, combined, and re-encoded. What SQL
buys is that the alignment is declarative and checkable -- a shape mismatch or
a missing tensor is a row in a result set, not an exception thrown halfway
through a merge that has already written half a file.

Tensors are processed in row-blocks rather than whole, so peak memory follows
the block size and not the model. On Python 3.11+, which has incremental blob
I/O, nothing model-sized is ever resident at all: merging a 70B checkpoint
costs the same as merging a 1B one, and both run in a couple of hundred
megabytes.

Four methods are implemented, all elementwise, so none of them needs the
stored shape:

    linear           weighted average of the models
    slerp            spherical interpolation between exactly two
    task-arithmetic  base + sum of scaled task vectors (m_i - base)
    ties             task vectors trimmed, sign-elected, then averaged

Only float dtypes can be merged. Averaging two Q4_K blobs byte by byte
produces noise that still parses as a model, which is the worst possible
failure, so quantized inputs are refused rather than approximated.
"""

import json
import math
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np

from reminis.db import open_for_bulk_copy
from reminis.backend import select as select_backend
from reminis.dtypes import is_float_dtype, to_float32, from_float32

METHODS = ("linear", "slerp", "task-arithmetic", "ties")

# SQLite's compile-time limit on ATTACH is 10 databases, and the base model
# for task arithmetic needs one of the slots.
MAX_INPUTS = 8

# Tensors are combined in blocks of this many elements: 4M floats is 16 MB
# decoded, small enough that a dozen of them at once is nothing and large
# enough that the per-block overhead disappears against the arithmetic.
CHUNK_ELEMENTS = 4 << 20

# Parameters for finding a trim threshold without holding the tensor. The
# histogram narrows the range that can contain the threshold; the limit is
# how many values may be collected for an exact ranking at the end.
TRIM_BINS = 4096
TRIM_REFINEMENTS = 4
TRIM_EXACT_LIMIT = 4 << 20


def merge_models(
    inputs: list[str],
    output_path: str,
    method: str = "linear",
    weights: list[float] | None = None,
    base: str | None = None,
    density: float = 0.2,
    t: float = 0.5,
    scale: float = 1.0,
    verbose: bool = True,
    backend: str | None = None,
) -> dict:
    """Merge model databases into a new one.

    Args:
        inputs: Paths to the model databases to merge, at least two.
        output_path: Where to write the merged model.
        method: One of ``linear``, ``slerp``, ``task-arithmetic``, ``ties``.
        weights: Per-input weights. Defaults to equal weights, and for
            ``linear`` they are normalised to sum to 1.
        base: The base model for ``task-arithmetic`` and ``ties``, i.e. the
            checkpoint every input was fine-tuned from.
        density: For ``ties``, the fraction of each task vector kept when
            trimming (0.2 keeps the largest 20% of entries).
        t: For ``slerp``, how far to travel from the first model to the
            second. 0 is the first model, 1 is the second.
        scale: Multiplier applied to the combined task vector before it is
            added back to the base (``task-arithmetic`` and ``ties``).
        verbose: Print a progress report.

    Returns:
        A summary dict describing what was merged.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown merge method '{method}'; expected one of {', '.join(METHODS)}")
    # One input is meaningful only against a base: base + scale * (m - base)
    # rescales a single fine-tune, and a negative scale subtracts it.
    if len(inputs) < 2 and not (base and method in ("task-arithmetic", "ties")):
        raise ValueError("Merging needs at least two models")
    if len(inputs) > MAX_INPUTS:
        raise ValueError(
            f"Merging is limited to {MAX_INPUTS} models at once "
            f"(SQLite attaches at most 10 databases to one connection)"
        )
    if method == "slerp" and len(inputs) != 2:
        raise ValueError("slerp interpolates between exactly two models")
    if method in ("task-arithmetic", "ties") and not base:
        raise ValueError(
            f"{method} works on task vectors (model - base), so it needs "
            f"--base pointing at the checkpoint these were fine-tuned from"
        )
    if not 0 < density <= 1:
        raise ValueError("--density must be in (0, 1]")

    paths = [Path(p) for p in inputs]
    for p in paths + ([Path(base)] if base else []):
        if not p.exists():
            raise FileNotFoundError(f"Database not found: {p}")

    # The output starts as a copy of the first model and every input is read
    # from while it is being written, so it cannot be one of them.
    out_resolved = Path(output_path).resolve()
    for p in paths + ([Path(base)] if base else []):
        if p.resolve() == out_resolved:
            raise ValueError(
                f"Output {output_path} is also an input; merging in place would "
                f"read from a file that is being rewritten. Choose another path."
            )

    if weights is None:
        weights = [1.0] * len(inputs)
    elif len(weights) != len(inputs):
        raise ValueError(
            f"Got {len(weights)} weights for {len(inputs)} models; they must match"
        )
    if method == "linear":
        total = sum(weights)
        if total == 0:
            raise ValueError("Linear merge weights sum to zero")
        weights = [w / total for w in weights]

    t0 = time.time()

    # Everything hangs off one connection: the output is a copy of the first
    # model (so metadata, tokenizer, and any tensor no one else has come
    # along), and every input is attached beside it.
    Path(output_path).unlink(missing_ok=True)
    shutil.copyfile(paths[0], output_path)
    conn = open_for_bulk_copy(output_path)

    aliases = []
    try:
        for i, p in enumerate(paths):
            alias = f"m{i}"
            conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(p.resolve()),))
            aliases.append(alias)
        if base:
            conn.execute("ATTACH DATABASE ? AS base", (str(Path(base).resolve()),))

        plan = _merge_plan(conn, aliases, use_base=bool(base))
        plan["coverage"] = _coverage(conn, plan)

        if verbose:
            _print_plan(paths, base, method, weights, plan)

        if plan["blocked"]:
            conn.close()
            Path(output_path).unlink(missing_ok=True)
            raise ValueError(_blocked_message(plan))

        # One transaction around the writes. The connection is in autocommit
        # so that ATTACH above is legal, which means without this every tensor
        # would commit on its own. Opened after the attaches for the same
        # reason: a database cannot be attached inside a transaction.
        conn.execute("BEGIN")
        summary = _apply_merge(
            conn, aliases, plan["mergeable"], method, weights,
            use_base=bool(base), density=density, t=t, scale=scale,
            verbose=verbose, backend=select_backend("elementwise", backend),
        )

        _record_provenance(conn, paths, base, method, weights, density, t, scale)
        conn.commit()
    except BaseException:
        # There is no journal to roll back, so a half-written merge would
        # otherwise be left on disk looking like a finished one. The output is
        # derived and reproducible; the inputs are untouched.
        conn.close()
        Path(output_path).unlink(missing_ok=True)
        raise
    finally:
        for alias in list(aliases) + (["base"] if base else []):
            try:
                conn.execute(f"DETACH DATABASE {alias}")
            except sqlite3.Error:
                pass
        conn.close()

    summary.update({
        "output": output_path,
        "method": method,
        "inputs": [str(p) for p in paths],
        "base": base,
        "weights": weights,
        "passthrough": plan["passthrough"],
        "unchanged": plan["unchanged"],
        "dropped": plan["dropped"],
        "seconds": time.time() - t0,
    })

    if verbose:
        _print_report(summary, output_path)

    return summary


def _merge_plan(conn: sqlite3.Connection, aliases: list[str], use_base: bool) -> dict:
    """Ask SQL which tensors line up across every attached model.

    Returns four disjoint sets: the tensors that can be merged, the ones only
    the first model has (which pass through untouched), the ones only later
    models have (dropped, since the result keeps the first model's structure),
    and the ones that appear everywhere but cannot be merged -- mismatched
    shapes, or dtypes that are not floats.
    """
    joins = " ".join(
        f"JOIN {a}.tensors t{i} ON t{i}.name = t0.name"
        for i, a in enumerate(aliases) if i > 0
    )
    if use_base:
        joins += " JOIN base.tensors tb ON tb.name = t0.name"

    others = [f"t{i}" for i in range(1, len(aliases))] + (["tb"] if use_base else [])
    shapes_match = " AND ".join(f"{o}.shape = t0.shape" for o in others) or "1"
    counts_match = " AND ".join(f"{o}.n_elements = t0.n_elements" for o in others) or "1"
    dtypes_differ = " OR ".join(f"{o}.dtype != t0.dtype" for o in others) or "0"

    rows = conn.execute(f"""
        SELECT t0.name,
               t0.dtype,
               t0.n_elements,
               ({shapes_match}) AND ({counts_match}) AS aligned,
               ({dtypes_differ})                      AS dtypes_differ
        FROM m0.tensors t0
        {joins}
        ORDER BY t0.id
    """).fetchall()

    present_everywhere = {r[0] for r in rows}

    others_data = " AND ".join(f"{o}.data = t0.data" for o in others) or "1"

    def identical_everywhere(name: str) -> bool:
        """Whether every model stores byte-identical data for this tensor.

        SQLite compares blobs directly, so this is one query rather than a
        read of every model's copy into Python. It is only asked about
        tensors that could not be merged numerically -- a non-float tensor
        that no model changed is not a conflict, it is just a constant.
        """
        row = conn.execute(f"""
            SELECT ({others_data}) FROM m0.tensors t0 {joins} WHERE t0.name = ?
        """, (name,)).fetchone()
        return bool(row and row[0])

    mergeable, blocked, unchanged = [], [], []
    for name, dtype, n_elements, aligned, dtypes_differ in rows:
        if not aligned:
            blocked.append((name, "shapes differ"))
        elif not is_float_dtype(dtype):
            if identical_everywhere(name):
                unchanged.append(name)
            else:
                blocked.append((name, f"{dtype} is quantized"))
        else:
            # A differing dtype between models is fine -- everything is
            # decoded to float32 anyway, and the result is written back in
            # the first model's dtype. It is worth reporting, not refusing.
            mergeable.append((name, dtype, n_elements, bool(dtypes_differ)))

    names_first = [r[0] for r in conn.execute("SELECT name FROM m0.tensors ORDER BY id")]
    passthrough = [n for n in names_first if n not in present_everywhere]

    dropped = set()
    for i in range(1, len(aliases)):
        for (n,) in conn.execute(f"SELECT name FROM m{i}.tensors"):
            if n not in present_everywhere and n not in set(names_first):
                dropped.add(n)

    return {
        "mergeable": mergeable,
        "blocked": blocked,
        "unchanged": unchanged,
        "passthrough": passthrough,
        "dropped": sorted(dropped),
    }


def _coverage(conn: sqlite3.Connection, plan: dict) -> float:
    """What fraction of the first model's weight bytes the merge actually touches.

    A quantized model whose only float tensors are its norms will merge
    happily and change almost nothing, because the quantized blocks are
    identical in both inputs or refused outright. That result is not wrong,
    but it is not a merge either, and it should say so.
    """
    total = conn.execute("SELECT SUM(n_bytes) FROM m0.tensors").fetchone()[0] or 0
    if not total:
        return 0.0
    names = [n for n, _, _, _ in plan["mergeable"]]
    merged = 0
    for i in range(0, len(names), 400):
        chunk = names[i:i + 400]
        placeholders = ",".join("?" * len(chunk))
        merged += conn.execute(
            f"SELECT SUM(n_bytes) FROM m0.tensors WHERE name IN ({placeholders})", chunk
        ).fetchone()[0] or 0
    return merged / total


class _TensorSlicer:
    """Reads one tensor out of one attached model, a block at a time.

    Python 3.11 gained incremental blob I/O, which reads a byte range out of
    a BLOB without materialising the rest of it -- exactly what is needed
    here, and the difference between a bounded merge and one that holds the
    whole tensor. Older Pythons fall back to fetching the blob once and
    slicing a memoryview of it, which still avoids decoding the whole thing
    to float32 but does keep the stored bytes around.

    SQLite's own `substr` is not a third option: it materialises the entire
    blob to answer each call, so asking for sixty blocks reads a 500 MB
    tensor sixty times.
    """

    def __init__(self, conn: sqlite3.Connection, alias: str, name: str, backend=None):
        rowid, dtype, n_elements, n_bytes = conn.execute(
            f"SELECT id, dtype, n_elements, n_bytes FROM {alias}.tensors WHERE name = ?",
            (name,),
        ).fetchone()
        self.backend = backend or select_backend("elementwise")
        self.dtype = dtype
        self.n_elements = n_elements
        self.itemsize = n_bytes // n_elements if n_elements else 1
        self._handle = None
        self._view = None

        if hasattr(conn, "blobopen"):
            try:
                self._handle = conn.blobopen(
                    "tensors", "data", rowid, readonly=True, name=alias
                )
            except (sqlite3.Error, AttributeError, TypeError):
                self._handle = None
        if self._handle is None:
            blob = conn.execute(
                f"SELECT data FROM {alias}.tensors WHERE name = ?", (name,)
            ).fetchone()[0]
            self._view = memoryview(blob)

    def _raw(self, start: int, count: int):
        lo, hi = start * self.itemsize, (start + count) * self.itemsize
        return self._handle[lo:hi] if self._handle is not None else self._view[lo:hi]

    def chunk(self, start: int, count: int):
        """A block in the backend's arrays, for the combining itself."""
        return self.backend.from_bytes(self._raw(start, count), self.dtype)

    def chunk_numpy(self, start: int, count: int) -> np.ndarray:
        """A block as plain numpy, for the passes that rank and bin values.

        Finding a trim threshold needs a histogram and an exact partial sort,
        which are numpy's to do; the arrays involved are one block at a time
        either way.
        """
        return to_float32(self._raw(start, count), self.dtype)

    def close(self):
        if self._handle is not None:
            self._handle.close()
        self._view = None


class _TensorWriter:
    """Writes a tensor into the output database a block at a time.

    The output starts as a copy of the first model, so the row already holds
    a blob of exactly the right size and incremental blob I/O can overwrite
    it in place -- no buffer holding the finished tensor at all. Without it,
    blocks accumulate in a bytearray and go in with one UPDATE.
    """

    def __init__(self, conn: sqlite3.Connection, name: str, n_elements: int,
                 dtype: str, backend=None):
        self.conn = conn
        self.name = name
        self.dtype = dtype
        self.backend = backend or select_backend("elementwise")
        row = conn.execute(
            "SELECT id, n_bytes FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        rowid, n_bytes = row
        self.itemsize = n_bytes // n_elements if n_elements else 1
        self._handle = None
        self._buffer = None

        if hasattr(conn, "blobopen"):
            try:
                self._handle = conn.blobopen("tensors", "data", rowid, readonly=False)
            except (sqlite3.Error, AttributeError, TypeError):
                self._handle = None
        if self._handle is None:
            self._buffer = bytearray()

    def write(self, start: int, values):
        raw = self.backend.to_bytes(values, self.dtype)
        if self._handle is not None:
            offset = start * self.itemsize
            self._handle[offset:offset + len(raw)] = raw
        else:
            self._buffer += raw

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        elif self._buffer is not None:
            self.conn.execute(
                "UPDATE tensors SET data = ?, n_bytes = ? WHERE name = ?",
                (bytes(self._buffer), len(self._buffer), self.name),
            )
            self._buffer = None


def _precompute(method, readers, base_reader, weights, density, n_elements):
    """Whole-tensor quantities that a single block cannot work out for itself.

    Linear and task arithmetic are purely elementwise and need nothing here.
    slerp needs the angle between the two models, which is two norms and a
    dot product. ties needs the magnitude threshold that trimming keeps,
    which is a rank statistic over every entry of each task vector.
    """
    if method == "linear" or method == "task-arithmetic":
        return None

    if method == "slerp":
        aa = bb = ab = 0.0
        for start in range(0, n_elements, CHUNK_ELEMENTS):
            count = min(CHUNK_ELEMENTS, n_elements - start)
            a = readers[0].chunk(start, count)
            b = readers[1].chunk(start, count)
            aa += float(a @ a)
            bb += float(b @ b)
            ab += float(a @ b)
        return {"na": math.sqrt(aa), "nb": math.sqrt(bb), "dot": ab}

    if method == "ties":
        return {"cutoffs": [
            _trim_cutoff(reader, base_reader, density, n_elements)
            for reader in readers
        ]}

    raise AssertionError(f"unreachable: {method}")


def _trim_cutoff(reader, base_reader, density, n_elements) -> float:
    """The magnitude at which trimming a task vector cuts, found in passes.

    Trimming keeps the largest `density` fraction of a task vector, which is
    a rank statistic: with the whole vector in memory it is one call to
    `np.partition`. Block by block it takes three passes -- the largest
    magnitude, then a histogram to find which narrow band the threshold
    falls in, then the values inside that band, which are few enough to sort
    exactly.

    The result is the same number `np.partition` would return, not an
    approximation of it, so a chunked ties merge and a whole-tensor one
    agree exactly. The band is narrowed repeatedly first, because a task
    vector where a huge number of entries share one magnitude would
    otherwise put them all in the collection pass at once.
    """
    if density >= 1:
        return float("-inf")
    k = max(1, int(round(n_elements * density)))
    if k >= n_elements:
        return float("-inf")

    def blocks():
        for start in range(0, n_elements, CHUNK_ELEMENTS):
            count = min(CHUNK_ELEMENTS, n_elements - start)
            yield np.abs(reader.chunk_numpy(start, count)
                         - base_reader.chunk_numpy(start, count))

    hi = 0.0
    for magnitudes in blocks():
        hi = max(hi, float(magnitudes.max(initial=0.0)))
    if hi == 0.0:
        return 0.0

    lo = 0.0
    # `above` counts entries already known to sit above the current band, so
    # the rank being sought inside the band is k - above.
    above = 0
    for _ in range(TRIM_REFINEMENTS):
        counts = np.zeros(TRIM_BINS, dtype=np.int64)
        for magnitudes in blocks():
            inside = magnitudes[(magnitudes >= lo) & (magnitudes <= hi)]
            counts += np.histogram(inside, bins=TRIM_BINS, range=(lo, hi))[0]

        edges = np.linspace(lo, hi, TRIM_BINS + 1)
        from_top = np.cumsum(counts[::-1])
        need = k - above
        crossing = int(np.searchsorted(from_top, need))
        if crossing >= TRIM_BINS:
            break
        bin_index = TRIM_BINS - 1 - crossing
        settled = int(from_top[crossing - 1]) if crossing > 0 else 0
        if int(counts[bin_index]) <= TRIM_EXACT_LIMIT:
            lo, hi, above = float(edges[bin_index]), float(edges[bin_index + 1]), settled
            break
        lo, hi, above = float(edges[bin_index]), float(edges[bin_index + 1]), settled

    candidates = []
    for magnitudes in blocks():
        candidates.append(magnitudes[(magnitudes >= lo) & (magnitudes <= hi)])
    band = np.concatenate(candidates) if candidates else np.zeros(0, dtype=np.float32)
    rank = k - above
    if rank <= 0:
        return float(hi)
    if rank > band.size:
        return float(lo)
    # The rank-th largest value inside the band is the threshold itself.
    return float(np.partition(band, band.size - rank)[band.size - rank])


def _blocked_message(plan: dict) -> str:
    quantized = [n for n, why in plan["blocked"] if "quantized" in why]
    shape = [n for n, why in plan["blocked"] if why == "shapes differ"]

    lines = ["Cannot merge these models."]
    if quantized:
        lines += [
            f"  {len(quantized)} tensors are quantized (e.g. {quantized[0]}), and "
            f"averaging quantized blocks byte by byte produces noise, not a model.",
            "  Merge the float weights instead: a safetensors checkpoint, or an "
            "F16/F32 GGUF.",
        ]
    if shape:
        lines += [
            f"  {len(shape)} tensors have different shapes across the inputs "
            f"(e.g. {shape[0]}).",
            "  These models are not the same architecture, so there is nothing "
            "elementwise to merge.",
        ]
    return "\n".join(lines)


def _apply_merge(
    conn, aliases, mergeable, method, weights, use_base, density, t, scale,
    verbose, backend=None
) -> dict:
    """Combine each aligned tensor and write it into the output.

    Every tensor is processed in row-blocks rather than whole, so peak memory
    is set by the block size instead of by the largest tensor in the model.
    That is what makes it possible to merge a model far bigger than RAM: a
    70B checkpoint's embedding matrix alone is a couple of gigabytes, and
    holding one decoded copy of it per input plus one for the output is the
    thing that would otherwise decide the memory ceiling.
    """
    backend = backend or select_backend("elementwise")
    xp = backend.xp

    n_params = 0
    mixed_dtype = 0
    drift_sum = 0.0
    drift_weight = 0.0
    max_drift = 0.0
    max_drift_name = ""

    for idx, (name, dtype, n_elements, dtypes_differ) in enumerate(mergeable):
        if dtypes_differ:
            mixed_dtype += 1

        readers = [_TensorSlicer(conn, alias, name, backend) for alias in aliases]
        base_reader = _TensorSlicer(conn, "base", name, backend) if use_base else None
        writer = _TensorWriter(conn, name, n_elements, dtype, backend)

        try:
            # slerp and ties both need a quantity computed over the whole
            # tensor before any block can be combined -- an angle for one, a
            # trim threshold for the other. Those get their own passes.
            precomputed = _precompute(method, readers, base_reader, weights,
                                      density, n_elements)

            delta_sq = 0.0
            ref_sq = 0.0
            for start in range(0, n_elements, CHUNK_ELEMENTS):
                count = min(CHUNK_ELEMENTS, n_elements - start)
                arrays = [r.chunk(start, count) for r in readers]
                base_arr = base_reader.chunk(start, count) if base_reader else None

                merged = _combine(method, arrays, weights, base_arr,
                                  density, t, scale, precomputed, start, xp)

                # Drift is measured against the first model, which is what
                # the output would have been had the merge done nothing.
                # Summing squares per block gives the same norms the
                # whole-tensor version computed.
                # Apple's Accelerate raises the divide-by-zero and overflow
                # flags during ordinary float32 dot products whose inputs and
                # results are all finite, so the warnings say nothing here.
                with backend.errstate():
                    diff = merged - arrays[0]
                    delta_sq += float(diff @ diff)
                    ref_sq += float(arrays[0] @ arrays[0])

                writer.write(start, merged)
        finally:
            for r in readers:
                r.close()
            if base_reader:
                base_reader.close()
            writer.close()

        if ref_sq > 0:
            rel = math.sqrt(delta_sq) / math.sqrt(ref_sq)
            drift_sum += rel * n_elements
            drift_weight += n_elements
            if rel > max_drift:
                max_drift, max_drift_name = rel, name

        n_params += n_elements

        if verbose and (idx + 1) % 50 == 0:
            print(f"  merged {idx + 1}/{len(mergeable)} tensors", end="\r", flush=True)

    if verbose and mergeable:
        print(" " * 40, end="\r")

    return {
        "tensors_merged": len(mergeable),
        "parameters_merged": n_params,
        "mixed_dtype": mixed_dtype,
        "mean_drift": drift_sum / drift_weight if drift_weight else 0.0,
        "max_drift": max_drift,
        "max_drift_tensor": max_drift_name,
    }


def _combine(method, arrays, weights, base_arr, density, t, scale,
             precomputed=None, start=0, xp=np):
    """Combine one block of the aligned tensors.

    `precomputed` carries whatever the method needed to know about the whole
    tensor before any block could be handled; it is None for the methods
    that are purely elementwise.
    """
    if method == "linear":
        out = weights[0] * arrays[0]
        for w, a in zip(weights[1:], arrays[1:]):
            out = out + w * a
        return out

    if method == "slerp":
        return _slerp(arrays[0], arrays[1], t, precomputed, xp)

    # Both remaining methods work on task vectors rather than the weights.
    taskvecs = [a - base_arr for a in arrays]

    if method == "task-arithmetic":
        combined = weights[0] * taskvecs[0]
        for w, tv in zip(weights[1:], taskvecs[1:]):
            combined = combined + w * tv
        return base_arr + scale * combined

    if method == "ties":
        cutoffs = precomputed["cutoffs"] if precomputed else None
        return base_arr + scale * _ties(taskvecs, weights, density, cutoffs, xp)

    raise AssertionError(f"unreachable: {method}")


def _slerp(a, b, t: float, stats=None, xp=np):
    """Spherical interpolation between two flattened weight tensors.

    The angle is measured between the normalised vectors but the
    interpolation is applied to the originals, so magnitude is carried along
    with direction. When the two are nearly parallel the sine denominator
    collapses and the formula loses all its precision, so that case falls
    back to a straight linear blend -- which is what slerp converges to
    anyway as the angle goes to zero.

    `stats` holds the two norms and the dot product measured over the whole
    tensor. They cannot be computed from a single block, since the angle
    between two vectors is not the angle between their first thousand
    entries, so a chunked merge works them out in a pass of its own first.
    """
    if stats is None:
        na, nb = float(xp.sqrt(a @ a)), float(xp.sqrt(b @ b))
        dot = float(a @ b)
    else:
        na, nb, dot = stats["na"], stats["nb"], stats["dot"]

    if na == 0 or nb == 0:
        return (1 - t) * a + t * b

    cos_omega = max(-1.0, min(1.0, dot / (na * nb)))
    omega = np.arccos(cos_omega)
    sin_omega = np.sin(omega)

    if abs(sin_omega) < 1e-6:
        return (1 - t) * a + t * b

    ka = float(np.sin((1 - t) * omega) / sin_omega)
    kb = float(np.sin(t * omega) / sin_omega)
    return ka * a + kb * b


def _ties(taskvecs, weights, density: float, cutoffs=None, xp=np):
    """TIES: trim each task vector, elect a sign, then average the agreers.

    Fine-tunes interfere in two ways -- they disagree about the direction of
    a parameter, and most of what they change is negligible. Trimming drops
    the negligible part; the sign election resolves the disagreements by
    letting total magnitude vote, so a parameter one model nudged and another
    shoved does not average out to nothing.

    Trimming is the one part that is not elementwise: which entries survive
    depends on how they rank against the whole task vector. `cutoffs` supplies
    those thresholds when the caller has already measured them across every
    block; without it, each vector is ranked here, which needs all of it.
    """
    trimmed = []
    for i, tv in enumerate(taskvecs):
        if density >= 1:
            trimmed.append(tv)
            continue
        if cutoffs is not None:
            cutoff = cutoffs[i]
        else:
            k = max(1, int(round(tv.size * density)))
            if k >= tv.size:
                trimmed.append(tv)
                continue
            # The k largest by magnitude survive; everything else becomes zero.
            cutoff = np.partition(np.abs(tv), tv.size - k)[tv.size - k]
        if cutoff == float("-inf"):
            trimmed.append(tv)
            continue
        trimmed.append(xp.where(xp.abs(tv) >= cutoff, tv, 0.0))

    elected = weights[0] * trimmed[0]
    for wi, tv in zip(weights[1:], trimmed[1:]):
        elected = elected + wi * tv
    sign = xp.sign(elected)

    numerator = None
    denominator = None
    for wi, tv in zip(weights, trimmed):
        agrees = (xp.sign(tv) == sign) & (tv != 0)
        num = xp.where(agrees, wi * tv, 0.0)
        den = xp.where(agrees, float(wi), 0.0)
        numerator = num if numerator is None else numerator + num
        denominator = den if denominator is None else denominator + den

    # Entries where no model agreed with the elected sign contribute nothing,
    # and dividing by their zero denominator would produce a NaN rather than
    # the zero that means "left alone".
    safe = xp.where(denominator != 0, denominator, 1.0)
    return xp.where(denominator != 0, numerator / safe, 0.0)


def _record_provenance(conn, paths, base, method, weights, density, t, scale):
    """Write down what this model is, so a merged file is never anonymous."""
    rows = [
        ("reminis.merge.method", method, "string"),
        ("reminis.merge.sources", json.dumps([p.name for p in paths]), "json"),
        ("reminis.merge.weights", json.dumps([round(w, 6) for w in weights]), "json"),
        ("reminis.merge.created", time.strftime("%Y-%m-%dT%H:%M:%S"), "string"),
    ]
    if base:
        rows.append(("reminis.merge.base", Path(base).name, "string"))
    if method == "slerp":
        rows.append(("reminis.merge.t", str(t), "string"))
    if method == "ties":
        rows.append(("reminis.merge.density", str(density), "string"))
    if method in ("ties", "task-arithmetic"):
        rows.append(("reminis.merge.scale", str(scale), "string"))

    name_row = conn.execute(
        "SELECT value FROM model_meta WHERE key = 'general.name'"
    ).fetchone()
    if name_row:
        rows.append(("general.name", f"{name_row[0]} ({method} merge)", "string"))

    conn.executemany(
        "INSERT OR REPLACE INTO model_meta (key, value, type) VALUES (?, ?, ?)", rows
    )


def _print_plan(paths, base, method, weights, plan):
    print(f"Merging {len(paths)} models with {method}")
    for p, w in zip(paths, weights):
        print(f"  {p.name:<44} weight {w:.4f}")
    if base:
        print(f"  base: {Path(base).name}")
    print()
    print(f"  {len(plan['mergeable'])} tensors align across every model")
    if plan["unchanged"]:
        print(f"  {len(plan['unchanged'])} are byte-identical in every model, left as they are")
    if plan["passthrough"]:
        print(f"  {len(plan['passthrough'])} only in {paths[0].name}, carried through unchanged")
    if plan["dropped"]:
        print(f"  {len(plan['dropped'])} only in later models, dropped "
              f"(the result keeps {paths[0].name}'s structure)")
    if plan["coverage"] < 0.9:
        print(f"\n  WARNING: the tensors being merged are only "
              f"{plan['coverage'] * 100:.1f}% of {paths[0].name}'s weight data.")
        print( "  The rest is quantized and identical in every input, so it is "
               "being carried")
        print( "  through untouched. The result is mostly the first model. "
               "Merge the float")
        print( "  weights instead if you want a real blend.")
    print()


def _print_report(s: dict, output_path: str):
    size = Path(output_path).stat().st_size
    print(f"Merged {s['tensors_merged']} tensors, {s['parameters_merged']:,} parameters")
    if s["mixed_dtype"]:
        print(f"  {s['mixed_dtype']} tensors had different dtypes across the inputs; "
              f"each was combined in float32 and written back in the first model's dtype")
    print(f"  Mean drift from {Path(s['inputs'][0]).name}: {s['mean_drift'] * 100:.2f}%")
    if s["max_drift_tensor"]:
        print(f"  Largest:  {s['max_drift'] * 100:.2f}%  ({s['max_drift_tensor']})")
    print(f"\nWrote {output_path} ({_fmt(size)}) in {s['seconds']:.1f}s")
    print(f"  Inspect it:  reminis info {output_path}")
    print(f"  Compare it:  reminis diff {s['inputs'][0]} {output_path}")


def _fmt(b: int) -> str:
    for unit, size in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if b >= size:
            return f"{b / size:.1f} {unit}"
    return f"{b} B"
