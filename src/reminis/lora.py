"""Turn a peft LoRA adapter into a reminis delta pack.

A LoRA adapter already *is* a low-rank delta: peft saves ``lora_A`` and
``lora_B``, and the update it applies to a base weight is ``scaling * B @ A``.
That is exactly the structure reminis v0.3.0 introduced for lossy packs, except
the factors here are the real ones rather than an SVD approximation of a
finished merge.

So an adapter converts to a delta pack with no SVD, no merge, and no
approximation error: the pack stores the adapter's own matrices verbatim in
float32. Applying it performs the merge, whose only inexactness is the
unavoidable rounding of the result back into the base tensor's dtype -- the
same rounding a peft ``merge_and_unload()`` does.

The practical point: a capability ships as a few MB of factors instead of a
full model copy, and `reminis apply` reproduces the merged model with a hash
you can check.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from reminis.diff import (
    DELTA_SCHEMA,
    _compress,
    _model_fingerprint,
    _reconstruction_hash,
    _version,
    _weights_hash,
)
from reminis.db import open_for_bulk_write, open_read_only
from reminis.dtypes import is_float_dtype
from reminis.safetensors_io import load_tensors

# peft writes `...q_proj.lora_A.weight`, and with a named adapter
# `...q_proj.lora_A.default.weight`. Both forms appear in the wild.
_LORA_SUFFIXES = ("lora_A", "lora_B")

# Wrapper prefixes peft adds around the original module path.
_WRAPPER_PREFIXES = ("base_model.model.", "base_model.")


def _split_lora_name(name: str):
    """Return (module_path, "lora_A"|"lora_B") or None if not a LoRA factor."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part in _LORA_SUFFIXES:
            return ".".join(parts[:i]), part
    return None


def _strip_wrappers(module: str) -> str:
    for prefix in _WRAPPER_PREFIXES:
        if module.startswith(prefix):
            return module[len(prefix) :]
    return module


def _candidate_base_names(module: str) -> list[str]:
    """Base-model tensor names a peft module path might correspond to.

    peft records the path inside its own wrapper, so the same weight can be
    spelled a few ways depending on how the model was loaded. Trying an
    ordered list beats guessing one and silently missing.
    """
    stripped = _strip_wrappers(module)
    candidates = [f"{stripped}.weight", f"{module}.weight"]
    if not stripped.startswith("model."):
        candidates.append(f"model.{stripped}.weight")
    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def _candidate_full_names(name: str) -> list[str]:
    """Base-model names for a full tensor peft saved via modules_to_save.

    Those are spelled `...embed_tokens.modules_to_save.default.weight`, so the
    wrapper segments have to come out before the name matches the base model.
    """
    parts = [p for p in name.split(".") if p not in ("original_module",)]
    if "modules_to_save" in parts:
        i = parts.index("modules_to_save")
        # Drop the marker and the adapter name that follows it.
        parts = parts[:i] + parts[i + 2 :]
    cleaned = ".".join(parts)
    stripped = _strip_wrappers(cleaned)
    candidates = [stripped, cleaned]
    if not stripped.startswith("model."):
        candidates.append(f"model.{stripped}")
    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def _read_adapter_config(adapter_path: Path) -> dict:
    """Load adapter_config.json from the adapter file's directory."""
    directory = adapter_path if adapter_path.is_dir() else adapter_path.parent
    config_path = directory / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json next to {adapter_path}. It carries the rank "
            "and alpha, and the delta has the wrong magnitude without them."
        )
    return json.loads(config_path.read_text())


def _scaling(config: dict) -> float:
    """The factor peft multiplies B @ A by before adding it to the weight.

    Getting this wrong does not fail -- it produces a delta of the wrong
    magnitude, which looks like a subtly worse model rather than a bug.
    """
    r = config.get("r")
    alpha = config.get("lora_alpha")
    if not r or alpha is None:
        raise ValueError("adapter_config.json is missing 'r' or 'lora_alpha'")
    if config.get("use_rslora"):
        # Rank-stabilised LoRA divides by sqrt(r) instead of r.
        return float(alpha) / float(np.sqrt(r))
    return float(alpha) / float(r)


def lora_to_delta_pack(
    adapter_path: str,
    base_db: str,
    output_path: str,
    verbose: bool = True,
) -> dict:
    """Convert a peft LoRA adapter into a delta pack against a base model.

    Args:
        adapter_path: An adapter_model.safetensors, or the directory holding it
            together with adapter_config.json.
        base_db: The base model database the adapter was trained against.
        output_path: Where to write the delta pack.
        verbose: Print a report.

    Returns:
        A summary dict.
    """
    adapter = Path(adapter_path)
    base_path = Path(base_db)
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter}")
    if not base_path.exists():
        raise FileNotFoundError(f"Base database not found: {base_path}")

    t0 = time.time()
    config = _read_adapter_config(adapter)
    scaling = _scaling(config)
    tensors = load_tensors(str(adapter))

    if config.get("fan_in_fan_out"):
        # Conv1D-style modules (GPT-2) store the weight transposed, so the
        # update must be transposed to match. Implemented but never run
        # against a real Conv1D checkpoint, so say so rather than imply it.
        if verbose:
            print("Note: fan_in_fan_out is set; the update will be transposed. "
                  "This path has not been validated against a real Conv1D model.")

    # Pair the A and B factors per module.
    pairs: dict[str, dict] = {}
    passthrough: dict[str, tuple] = {}
    unsupported = []

    for name, (dtype, shape, blob) in tensors.items():
        if "lora_embedding_" in name:
            unsupported.append(name)
            continue
        split = _split_lora_name(name)
        if split is None:
            # modules_to_save entries: full tensors peft trained outright
            # (often the embedding or the classifier head).
            passthrough[name] = (dtype, shape, blob)
            continue
        module, which = split
        pairs.setdefault(module, {})[which] = (dtype, shape, blob)

    if unsupported:
        raise ValueError(
            "This adapter contains embedding LoRA factors "
            f"({', '.join(sorted(unsupported)[:3])}...), which reminis does not "
            "handle yet. Refusing rather than writing a pack missing part of "
            "the adapter."
        )

    incomplete = sorted(m for m, p in pairs.items() if len(p) != 2)
    if incomplete:
        raise ValueError(
            f"Adapter has unpaired LoRA factors for: {', '.join(incomplete[:5])}"
        )

    base_conn = open_read_only(str(base_path))
    base_names = {r[0] for r in base_conn.execute("SELECT name FROM tensors")}

    # A pack written from nothing into a path just unlinked, so it opens the
    # way every other fresh file does. This wrote through WAL until 0.32.2,
    # having missed the change delta packs got in 0.29.1.
    Path(output_path).unlink(missing_ok=True)
    out_conn = open_for_bulk_write(output_path)
    out_conn.executescript(DELTA_SCHEMA)
    out_conn.execute("BEGIN")

    unmatched = []
    written = 0
    total_raw = 0
    total_stored = 0
    ranks = []

    for module in sorted(pairs):
        target = next(
            (c for c in _candidate_base_names(module) if c in base_names), None
        )
        if target is None:
            unmatched.append(module)
            continue

        shape_json, dtype, dtype_id, n_elements, n_bytes = base_conn.execute(
            "SELECT shape, dtype, dtype_id, n_elements, n_bytes FROM tensors WHERE name = ?",
            (target,),
        ).fetchone()

        if not is_float_dtype(dtype):
            raise ValueError(
                f"Base tensor '{target}' is {dtype}, which cannot be merged with a "
                "LoRA update. Quantized base models must be dequantized first."
            )

        a_dtype, a_shape, a_blob = pairs[module]["lora_A"]
        b_dtype, b_shape, b_blob = pairs[module]["lora_B"]

        a = _as_f32(a_blob, a_dtype).reshape(a_shape)  # (r, in)
        b = _as_f32(b_blob, b_dtype).reshape(b_shape)  # (out, r)
        rank = a_shape[0]
        if b_shape[1] != rank:
            raise ValueError(
                f"Rank mismatch for {module}: lora_A is {a_shape}, lora_B is {b_shape}"
            )

        # Fold the scaling into B so apply is a plain matmul, as the pack
        # format expects.
        left, right = (b * scaling), a
        if config.get("fan_in_fan_out"):
            left, right = right.T.copy(), left.T.copy()

        # shape is stored reversed, so (m, n) is the data layout.
        stored_shape = json.loads(shape_json)
        m, n = stored_shape[1], stored_shape[0]
        if left.shape != (m, rank) or right.shape != (rank, n):
            raise ValueError(
                f"Adapter factors for {module} do not fit base tensor '{target}': "
                f"factors give {left.shape} @ {right.shape}, base is ({m}, {n})"
            )

        # float32 factors, not float16: these are the adapter's real numbers,
        # and the whole point of this path is that nothing is approximated.
        header = json.dumps({"rank": rank, "m": m, "n": n, "fdt": "f32"}).encode()
        payload = _compress(
            len(header).to_bytes(4, "little")
            + header
            + left.astype(np.float32).tobytes()
            + right.astype(np.float32).tobytes()
        )

        out_conn.execute(
            "INSERT INTO deltas (tensor_name, shape, dtype, dtype_id, n_elements, "
            "encoding, raw_bytes, stored_bytes, rank, data) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (target, shape_json, dtype, dtype_id, n_elements, "lowrank_zstd",
             n_bytes, len(payload), rank, payload),
        )
        written += 1
        total_raw += n_bytes
        total_stored += len(payload)
        ranks.append(rank)

    # modules_to_save tensors replace their base counterpart outright.
    replaced = 0
    for name, (dtype, shape, blob) in sorted(passthrough.items()):
        target = next(
            (c for c in _candidate_full_names(name) if c in base_names), None
        )
        if target is None:
            unmatched.append(name)
            continue
        dtype_id, n_elements = base_conn.execute(
            "SELECT dtype_id, n_elements FROM tensors WHERE name = ?", (target,)
        ).fetchone()
        payload = _compress(blob)
        out_conn.execute(
            "INSERT INTO deltas (tensor_name, shape, dtype, dtype_id, n_elements, "
            "encoding, raw_bytes, stored_bytes, data) VALUES (?,?,?,?,?,?,?,?)",
            (target, json.dumps(shape[::-1]), dtype, dtype_id, n_elements,
             "replace_zstd", len(blob), len(payload), payload),
        )
        replaced += 1
        total_raw += len(blob)
        total_stored += len(payload)

    if unmatched:
        out_conn.close()
        base_conn.close()
        Path(output_path).unlink(missing_ok=True)
        raise ValueError(
            "These adapter modules have no matching tensor in the base model:\n  "
            + "\n  ".join(sorted(unmatched)[:10])
            + "\nThe adapter was probably trained against a different base, or the "
            "base was converted from GGUF (whose tensor names differ from PyTorch's)."
        )

    if written == 0 and replaced == 0:
        out_conn.close()
        base_conn.close()
        Path(output_path).unlink(missing_ok=True)
        raise ValueError(f"No usable tensors found in {adapter}")

    out_conn.commit()
    out_conn.execute("BEGIN")

    fingerprint = _model_fingerprint(base_conn)
    meta = {
        "base_fingerprint": fingerprint,
        # A LoRA merge changes values, never names or shapes, so the structure
        # of the result is the base's -- except where modules_to_save replaced
        # a tensor, which also keeps its shape.
        "target_fingerprint": fingerprint,
        "base_weights_hash": _weights_hash(base_conn),
        "base_file": base_path.name,
        "target_file": f"{base_path.stem}+{adapter.name}",
        "tensors_changed": str(written + replaced),
        "tensors_identical": "0",
        "tensors_removed": "[]",
        # No approximation is introduced: these are the adapter's own factors,
        # stored exactly. The pack reproduces its target bit-for-bit.
        "lossy": "false",
        "lowrank_tensors": str(written),
        "max_rel_error": "0.0",
        "source": "lora_adapter",
        "lora_r": str(config.get("r")),
        "lora_alpha": str(config.get("lora_alpha")),
        "lora_scaling": repr(scaling),
        "lora_use_rslora": "true" if config.get("use_rslora") else "false",
        "adapter_file": adapter.name,
        "reminis_version": _version(),
    }

    target_hash = _reconstruction_hash(base_conn, out_conn, [])
    meta["target_weights_hash"] = target_hash
    meta["reconstructed_weights_hash"] = target_hash

    for k, v in meta.items():
        out_conn.execute(
            "INSERT OR REPLACE INTO delta_meta (key, value) VALUES (?,?)", (k, v)
        )
    out_conn.commit()
    out_conn.close()

    base_total = base_conn.execute("SELECT SUM(n_bytes) FROM tensors").fetchone()[0] or 0
    base_conn.close()

    summary = {
        "adapter": str(adapter),
        "base": str(base_path),
        "pack": output_path,
        "lowrank_tensors": written,
        "replaced_tensors": replaced,
        "ranks": sorted(set(ranks)),
        "scaling": scaling,
        "base_total_bytes": base_total,
        "pack_raw_bytes": total_raw,
        "pack_stored_bytes": total_stored,
        "target_weights_hash": target_hash,
        "elapsed": time.time() - t0,
    }

    if verbose:
        _print_report(summary, config)

    return summary


def _as_f32(blob: bytes, dtype: str) -> np.ndarray:
    from reminis.dtypes import to_float32

    return to_float32(blob, dtype)


def _print_report(s: dict, config: dict):
    from reminis.diff import _fmt_bytes

    ranks = ", ".join(str(r) for r in s["ranks"])
    print(f"\nAdapter: {Path(s['adapter']).name}")
    print(f"  rank {ranks}, alpha {config.get('lora_alpha')}, scaling {s['scaling']:.4g}")
    print(f"  {s['lowrank_tensors']} tensors as exact low-rank factors")
    if s["replaced_tensors"]:
        print(f"  {s['replaced_tensors']} tensors replaced in full (modules_to_save)")
    print(f"\n  Base model:  {_fmt_bytes(s['base_total_bytes'])}")
    print(f"  Delta pack:  {_fmt_bytes(s['pack_stored_bytes'])}")
    if s["base_total_bytes"]:
        pct = s["pack_stored_bytes"] / s["base_total_bytes"] * 100
        print(f"  Ratio:       {pct:.2f}% of the base model")
    print("  Encoding:    exact low-rank (the adapter's own factors, float32)")
    print(f"\n  Took {s['elapsed']:.1f}s")
