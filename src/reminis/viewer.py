"""Generate a self-contained HTML viewer for a reminis database."""

import json
import re
import sqlite3
import struct
import webbrowser
from importlib import resources
from pathlib import Path

import numpy as np

from .db import SELECT_TENSOR_ROWS
from .dtypes import is_float_dtype, to_float32

# The two values viewer_template.html leaves open. Spelled out rather than
# built from the keys below so that a typo in one place fails to match
# rather than silently substituting nothing.
_PLACEHOLDER = re.compile(r"__REMINIS_MODEL_NAME__|__REMINIS_DATA__")


def _compute_tensor_stats(data_blob: bytes, dtype_name: str, n_elements: int) -> dict:
    """Compute basic statistics for a tensor.

    Decoding goes through ``dtypes.to_float32`` rather than a local
    ``np.frombuffer``: BF16 and F16 are both 16 bits, so reading one as the
    other is not an error, it is silently wrong arithmetic. This function read
    BF16 as float16 through 0.32.1 and reported 3.39 for a stored 100.0.
    """
    try:
        if not is_float_dtype(dtype_name):
            return {"quantized": True, "n_bytes": len(data_blob)}
        arr = to_float32(data_blob, dtype_name)

        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "abs_mean": float(np.mean(np.abs(arr))),
            "zeros_pct": float(np.sum(arr == 0) / len(arr) * 100),
            "n_elements": len(arr),
        }
    except Exception as exc:
        # Still caught: a viewer over a few hundred tensors should not fail to
        # render because one of them is odd. But the reason is kept rather than
        # flattened to `True`, so a systematic decoding fault shows up as the
        # same message on every tensor instead of a blank panel that looks
        # like a property of the model.
        return {"error": f"{type(exc).__name__}: {exc}", "n_bytes": len(data_blob)}


def _sample_heatmap(data_blob: bytes, dtype_name: str, shape: list, size: int = 48) -> list | None:
    """Sample a 2D heatmap from tensor data."""
    try:
        if not is_float_dtype(dtype_name):
            return None
        arr = to_float32(data_blob, dtype_name)

        if len(shape) < 2:
            return None

        rows, cols = shape[-2], shape[-1]
        arr = arr[:rows * cols].reshape(rows, cols)

        row_idx = np.linspace(0, rows - 1, min(size, rows), dtype=int)
        col_idx = np.linspace(0, cols - 1, min(size, cols), dtype=int)
        sampled = arr[np.ix_(row_idx, col_idx)]

        vmax = float(np.max(np.abs(sampled)))
        if vmax > 0:
            sampled = sampled / vmax

        return [[round(float(v), 3) for v in row] for row in sampled]
    except Exception:
        # No error channel here on purpose: None already means "no heatmap for
        # this tensor", which is also the answer for a quantized or 1-D one.
        # Anything that breaks the decode breaks it in _compute_tensor_stats
        # too, and that is where the reason gets reported.
        return None


def generate_viewer(db_path: str, output_path: str | None = None, verbose: bool = True) -> str:
    """Generate a self-contained HTML viewer for a reminis database."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    if output_path is None:
        output_path = str(db_path.with_suffix(".html"))

    conn = sqlite3.connect(str(db_path))

    # Gather metadata
    meta = {}
    for key, value, type_name in conn.execute("SELECT key, value, type FROM model_meta"):
        meta[key] = {"value": value, "type": type_name}

    # Gather tensor info + stats
    tensors = []
    rows = conn.execute(SELECT_TENSOR_ROWS).fetchall()

    for name, shape_str, dtype_name, dtype_id, n_elements, n_bytes, data_blob in rows:
        shape = json.loads(shape_str)
        stats = _compute_tensor_stats(data_blob, dtype_name, n_elements)
        heatmap = _sample_heatmap(data_blob, dtype_name, shape)

        tensors.append({
            "name": name,
            "shape": shape,
            "dtype": dtype_name,
            "n_elements": n_elements,
            "n_bytes": n_bytes,
            "stats": stats,
            "heatmap": heatmap,
        })

    conn.close()

    # Summary stats
    total_params = sum(t["n_elements"] for t in tensors)
    total_bytes = sum(t["n_bytes"] for t in tensors)
    dtype_counts = {}
    for t in tensors:
        d = t["dtype"]
        if d not in dtype_counts:
            dtype_counts[d] = {"count": 0, "bytes": 0, "params": 0}
        dtype_counts[d]["count"] += 1
        dtype_counts[d]["bytes"] += t["n_bytes"]
        dtype_counts[d]["params"] += t["n_elements"]

    model_name = meta.get("general.name", {}).get("value", db_path.stem)
    arch = meta.get("general.architecture", {}).get("value", "unknown")

    viewer_data = {
        "model_name": model_name,
        "architecture": arch,
        "db_file": db_path.name,
        "db_size_mb": round(db_path.stat().st_size / (1024 * 1024), 1),
        "total_params": total_params,
        "total_bytes": total_bytes,
        "dtype_counts": dtype_counts,
        "meta": {k: v["value"] for k, v in meta.items()},
        "tensors": tensors,
    }

    html = _build_html(viewer_data)

    with open(output_path, "w") as f:
        f.write(html)

    if verbose:
        size_mb = len(html) / (1024 * 1024)
        print(f"Viewer written to {output_path} ({size_mb:.1f} MB)")

    return output_path


def _build_html(data: dict) -> str:
    """Fill the viewer template with this model's data.

    The template is a real .html file rather than an f-string in here: it
    is twelve hundred lines of CSS and JavaScript, and as an f-string every
    brace in it had to be doubled, which no editor highlights and no linter
    checks. The two things that vary are substituted by name.

    re.sub does the substitution in a single left-to-right pass, so a model
    name or a blob of data that happens to contain one of the placeholders
    is inserted, not rescanned.
    """
    values = {
        "__REMINIS_MODEL_NAME__": str(data["model_name"]),
        "__REMINIS_DATA__": json.dumps(data, separators=(",", ":")),
    }
    template = resources.files("reminis").joinpath("viewer_template.html").read_text(
        encoding="utf-8"
    )
    return _PLACEHOLDER.sub(lambda m: values[m.group(0)], template)
