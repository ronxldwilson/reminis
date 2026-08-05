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
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np

from reminis.dtypes import is_float_dtype, to_float32, from_float32

METHODS = ("linear", "slerp", "task-arithmetic", "ties")

# SQLite's compile-time limit on ATTACH is 10 databases, and the base model
# for task arithmetic needs one of the slots.
MAX_INPUTS = 8


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
    conn = sqlite3.connect(output_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

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

        summary = _apply_merge(
            conn, aliases, plan["mergeable"], method, weights,
            use_base=bool(base), density=density, t=t, scale=scale,
            verbose=verbose,
        )

        _record_provenance(conn, paths, base, method, weights, density, t, scale)
        conn.commit()
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
    conn, aliases, mergeable, method, weights, use_base, density, t, scale, verbose
) -> dict:
    """Combine each aligned tensor and write it into the output."""
    n_params = 0
    mixed_dtype = 0
    drift_sum = 0.0
    drift_weight = 0.0
    max_drift = 0.0
    max_drift_name = ""

    for idx, (name, dtype, n_elements, dtypes_differ) in enumerate(mergeable):
        if dtypes_differ:
            mixed_dtype += 1

        arrays = []
        for i, alias in enumerate(aliases):
            blob, d = conn.execute(
                f"SELECT data, dtype FROM {alias}.tensors WHERE name = ?", (name,)
            ).fetchone()
            arrays.append(to_float32(blob, d))

        base_arr = None
        if use_base:
            blob, d = conn.execute(
                "SELECT data, dtype FROM base.tensors WHERE name = ?", (name,)
            ).fetchone()
            base_arr = to_float32(blob, d)

        merged = _combine(method, arrays, weights, base_arr, density, t, scale)

        # Drift is measured against the first model, which is what the output
        # would have been if the merge had done nothing.
        delta = merged - arrays[0]
        ref = float(np.linalg.norm(arrays[0]))
        if ref > 0:
            rel = float(np.linalg.norm(delta)) / ref
            drift_sum += rel * n_elements
            drift_weight += n_elements
            if rel > max_drift:
                max_drift, max_drift_name = rel, name

        out = from_float32(merged, dtype)
        conn.execute(
            "UPDATE tensors SET data = ?, n_bytes = ? WHERE name = ?",
            (out, len(out), name),
        )
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


def _combine(method, arrays, weights, base_arr, density, t, scale) -> np.ndarray:
    if method == "linear":
        out = np.zeros_like(arrays[0])
        for w, a in zip(weights, arrays):
            out += np.float32(w) * a
        return out

    if method == "slerp":
        return _slerp(arrays[0], arrays[1], t)

    # Both remaining methods work on task vectors rather than the weights.
    taskvecs = [a - base_arr for a in arrays]

    if method == "task-arithmetic":
        combined = np.zeros_like(base_arr)
        for w, tv in zip(weights, taskvecs):
            combined += np.float32(w) * tv
        return base_arr + np.float32(scale) * combined

    if method == "ties":
        return base_arr + np.float32(scale) * _ties(taskvecs, weights, density)

    raise AssertionError(f"unreachable: {method}")


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two flattened weight tensors.

    The angle is measured between the normalised vectors but the
    interpolation is applied to the originals, so magnitude is carried along
    with direction. When the two are nearly parallel the sine denominator
    collapses and the formula loses all its precision, so that case falls
    back to a straight linear blend -- which is what slerp converges to
    anyway as the angle goes to zero.
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return (1 - np.float32(t)) * a + np.float32(t) * b

    cos_omega = float(np.dot(a, b) / (na * nb))
    cos_omega = max(-1.0, min(1.0, cos_omega))
    omega = np.arccos(cos_omega)
    sin_omega = np.sin(omega)

    if abs(sin_omega) < 1e-6:
        return (1 - np.float32(t)) * a + np.float32(t) * b

    ka = np.float32(np.sin((1 - t) * omega) / sin_omega)
    kb = np.float32(np.sin(t * omega) / sin_omega)
    return ka * a + kb * b


def _ties(taskvecs: list[np.ndarray], weights: list[float], density: float) -> np.ndarray:
    """TIES: trim each task vector, elect a sign, then average the agreers.

    Fine-tunes interfere in two ways -- they disagree about the direction of
    a parameter, and most of what they change is negligible. Trimming drops
    the negligible part; the sign election resolves the disagreements by
    letting total magnitude vote, so a parameter one model nudged and another
    shoved does not average out to nothing.
    """
    trimmed = []
    for tv in taskvecs:
        if density >= 1:
            trimmed.append(tv)
            continue
        k = max(1, int(round(tv.size * density)))
        if k >= tv.size:
            trimmed.append(tv)
            continue
        # The k largest by magnitude survive; everything else becomes zero.
        cutoff = np.partition(np.abs(tv), tv.size - k)[tv.size - k]
        trimmed.append(np.where(np.abs(tv) >= cutoff, tv, np.float32(0)))

    w = [np.float32(x) for x in weights]

    elected = np.zeros_like(trimmed[0])
    for wi, tv in zip(w, trimmed):
        elected += wi * tv
    sign = np.sign(elected)

    numerator = np.zeros_like(elected)
    denominator = np.zeros_like(elected)
    for wi, tv in zip(w, trimmed):
        agrees = (np.sign(tv) == sign) & (tv != 0)
        numerator += np.where(agrees, wi * tv, np.float32(0))
        denominator += np.where(agrees, wi, np.float32(0))

    return np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator), where=denominator != 0,
    )


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
