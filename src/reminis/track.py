"""Record how a model got the way it is: an edit log for training runs.

Everything else in reminis compares models that already exist. This closes the
loop by logging every optimizer step as it happens -- gradient magnitudes,
weight norms, loss -- so a bad step can be found afterwards with a SQL query
instead of a guess.

Two things are recorded, and they answer different questions:

* **Statistics** (cheap, always on): per-parameter gradient and weight norms
  per step. Enough to *locate* a bad update. A few hundred small numbers per
  step, so it stays affordable on a real run.
* **Snapshots** (expensive, occasional): the actual weights, stored as delta
  packs against the previous snapshot so a long run does not cost one full
  model copy per checkpoint. Enough to *undo* back to a known state.

What this deliberately does not do is store every step's full weight delta.
For a 135M-parameter model that is hundreds of MB per step, and the updates
are high-entropy float mantissas that do not compress. Trajectory storage is
practical only for the trainable subset of a LoRA-style run, which is what
`track_params` is for.

See `rollback_to_step` for what rollback actually delivers, which is narrower
than the word suggests.

torch is never imported here. Arrays arrive as anything exposing
`detach`/`cpu`/`numpy` or the buffer protocol, so this works under plain
PyTorch, HuggingFace `Trainer`, or a hand-written loop.
"""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
    step            INTEGER PRIMARY KEY,
    epoch           REAL,
    loss            REAL,
    learning_rate   REAL,
    grad_norm_total REAL,
    wall_time       REAL,
    logged_at       REAL
);

CREATE TABLE IF NOT EXISTS param_updates (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    step               INTEGER NOT NULL,
    param              TEXT NOT NULL,
    grad_norm          REAL,
    grad_mean          REAL,
    grad_max           REAL,
    weight_norm_before REAL,
    weight_norm_after  REAL,
    update_norm        REAL,
    rolled_back        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS snapshots (
    step         INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL,
    base_step    INTEGER,
    path         TEXT NOT NULL,
    weights_hash TEXT NOT NULL,
    bytes        INTEGER NOT NULL,
    created_at   REAL
);

CREATE INDEX IF NOT EXISTS idx_updates_step  ON param_updates(step);
CREATE INDEX IF NOT EXISTS idx_updates_param ON param_updates(param);
"""


def _to_numpy(x) -> np.ndarray:
    """Get a numpy view of a tensor without importing the framework."""
    for attr in ("detach", "cpu"):
        if hasattr(x, attr):
            x = getattr(x, attr)()
    if hasattr(x, "float"):
        # bfloat16 has no numpy equivalent; widen before crossing over.
        try:
            if str(getattr(x, "dtype", "")) in ("torch.bfloat16", "torch.float16"):
                x = x.float()
        except Exception:
            pass
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _tensor_stats(x) -> tuple[float, float, float]:
    """(l2 norm, mean, max abs), reduced in the framework where possible.

    Converting to numpy first means copying the tensor off the device and
    walking it several times on one CPU thread. Reducing in place instead
    keeps the work where the tensor already lives -- multithreaded on CPU, and
    essentially free on a GPU, which is where real training happens. Only three
    scalars ever cross over.

    Falls back to numpy for anything that is not a framework tensor.
    """
    if hasattr(x, "norm") and hasattr(x, "aminmax") and hasattr(x, "numel"):
        if x.numel() == 0:
            return 0.0, 0.0, 0.0
        flat = x.detach().reshape(-1).float()
        lo, hi = flat.aminmax()
        return (
            float(flat.norm()),
            float(flat.mean()),
            max(abs(float(lo)), abs(float(hi))),
        )
    return _stats(_to_numpy(x))


def _stats(arr: np.ndarray) -> tuple[float, float, float]:
    """(l2 norm, mean, max abs) of a tensor.

    Written to allocate nothing. The obvious implementations all copy the whole
    tensor -- ``astype(np.float64)`` on a 113M-element embedding is a 900 MB
    allocation, and ``np.abs(x).max()`` is another full-size temporary. Doing
    that per parameter per step doubled step time on a 135M model.

    So: ``norm`` goes to BLAS on the native dtype, ``mean`` uses a float64
    accumulator without widening the input, and max-abs comes from the two
    one-pass extremes instead of materialising ``abs``.
    """
    flat = arr.reshape(-1)
    if flat.size == 0:
        return 0.0, 0.0, 0.0
    lo = float(flat.min())
    hi = float(flat.max())
    return (
        float(np.linalg.norm(flat)),
        float(flat.mean(dtype=np.float64)),
        max(abs(lo), abs(hi)),
    )


class TrainingLog:
    """An append-only log of what a training run did to a model's weights."""

    def __init__(
        self,
        path: str,
        run_name: str | None = None,
        snapshot_dir: str | None = None,
    ):
        self.path = str(path)
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else Path(self.path).parent
        self.conn = sqlite3.connect(self.path)
        # WAL plus a relaxed sync: this is a diagnostic log written inside a
        # training loop, and the default fsync-per-commit cost would show up
        # directly in step time.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(LOG_SCHEMA)

        self._prev_norms: dict[str, float] = {}
        self._t0 = time.time()
        self._last_snapshot_step: int | None = None

        # INSERT OR IGNORE, not REPLACE: opening an existing log to read it --
        # which `reminis log` does -- must not overwrite the run's identity or
        # its start time with fresh values.
        meta = {
            "run_name": run_name or f"run-{int(self._t0)}",
            "started_at": repr(self._t0),
            "reminis_version": _version(),
        }
        for k, v in meta.items():
            self.conn.execute(
                "INSERT OR IGNORE INTO run_meta (key, value) VALUES (?,?)", (k, str(v))
            )
        if run_name is not None:
            self.conn.execute(
                "INSERT OR REPLACE INTO run_meta (key, value) VALUES ('run_name', ?)",
                (run_name,),
            )
        # Resume where a previous session left off rather than re-deriving
        # snapshot bases from an empty table.
        row = self.conn.execute("SELECT MAX(step) FROM snapshots").fetchone()
        if row and row[0] is not None:
            self._last_snapshot_step = int(row[0])
        self.conn.commit()

    # -- writing -----------------------------------------------------------

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            self.conn.execute(
                "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?,?)",
                (k, json.dumps(v) if not isinstance(v, str) else v),
            )
        self.conn.commit()

    def log_step(
        self,
        step: int,
        named_grads: dict,
        named_weights: dict | None = None,
        loss: float | None = None,
        epoch: float | None = None,
        learning_rate: float | None = None,
    ):
        """Record one optimizer step.

        Args:
            step: Global step number.
            named_grads: {param name: gradient tensor}, read *before* the
                gradients are zeroed.
            named_weights: {param name: weight tensor} after the step. Weight
                norms before the step are carried over from the previous step
                rather than measured again -- weights only change here, so the
                two are the same number and measuring twice doubles the cost.
            loss: Training loss at this step.
            epoch: Fractional epoch.
            learning_rate: LR in effect for this step.
        """
        rows = []
        total_sq = 0.0

        for name, grad in named_grads.items():
            if grad is None:
                continue
            g_norm, g_mean, g_max = _tensor_stats(grad)
            total_sq += g_norm ** 2

            before = self._prev_norms.get(name)
            after = None
            update_norm = None
            if named_weights is not None and name in named_weights:
                after, _, _ = _tensor_stats(named_weights[name])
                self._prev_norms[name] = after

            rows.append(
                (step, name, g_norm, g_mean, g_max, before, after, update_norm)
            )

        self.conn.executemany(
            "INSERT INTO param_updates (step, param, grad_norm, grad_mean, grad_max, "
            "weight_norm_before, weight_norm_after, update_norm) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO steps (step, epoch, loss, learning_rate, "
            "grad_norm_total, wall_time, logged_at) VALUES (?,?,?,?,?,?,?)",
            (
                step, epoch, loss, learning_rate,
                float(np.sqrt(total_sq)), time.time() - self._t0, time.time(),
            ),
        )
        self.conn.commit()
        return len(rows)

    def snapshot(self, step: int, state_dict: dict, delta: bool = True) -> dict:
        """Store the model's weights at this step.

        By default this is written as a delta pack against the previous
        snapshot, so a run with many checkpoints does not cost one full model
        copy each time. The first snapshot is necessarily full.

        Returns a dict describing what was written.
        """
        from reminis.diff import _weights_hash, diff_models

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        full_path = self.snapshot_dir / f"snapshot-{step:06d}.db"
        state_dict_to_sqlite(state_dict, str(full_path))

        conn = sqlite3.connect(str(full_path))
        weights_hash = _weights_hash(conn)
        conn.close()

        kind, base_step, stored_path = "full", None, full_path

        # A snapshot can only be a delta against an *earlier* one. Without this
        # guard, re-snapshotting the same step makes the row its own base and
        # the delta chain becomes a cycle.
        if delta and self._last_snapshot_step is not None and self._last_snapshot_step < step:
            prev = self.conn.execute(
                "SELECT path, kind FROM snapshots WHERE step = ?",
                (self._last_snapshot_step,),
            ).fetchone()
            prev_full = self._materialise(self._last_snapshot_step) if prev else None
            if prev_full is not None:
                pack_path = self.snapshot_dir / f"snapshot-{step:06d}.pack.db"
                diff_models(str(prev_full), str(full_path), str(pack_path), verbose=False)
                # Only keep the pack if it actually beat the full copy. Early
                # in training almost every weight moves, and a pack of a
                # fully-changed model is not smaller.
                if pack_path.stat().st_size < full_path.stat().st_size:
                    full_path.unlink()
                    kind, base_step, stored_path = "delta", self._last_snapshot_step, pack_path
                else:
                    pack_path.unlink()

        size = stored_path.stat().st_size
        self.conn.execute(
            "INSERT OR REPLACE INTO snapshots (step, kind, base_step, path, "
            "weights_hash, bytes, created_at) VALUES (?,?,?,?,?,?,?)",
            (step, kind, base_step, str(stored_path), weights_hash, size, time.time()),
        )
        self.conn.commit()
        self._last_snapshot_step = step

        return {
            "step": step, "kind": kind, "base_step": base_step,
            "path": str(stored_path), "bytes": size, "weights_hash": weights_hash,
        }

    def _materialise(self, step: int) -> Path | None:
        """Reconstruct a snapshot's full database, following the delta chain."""
        from reminis.diff import apply_delta

        row = self.conn.execute(
            "SELECT kind, base_step, path FROM snapshots WHERE step = ?", (step,)
        ).fetchone()
        if row is None:
            return None
        kind, base_step, path = row
        if kind == "full":
            return Path(path) if Path(path).exists() else None

        if base_step is None or base_step >= step:
            raise ValueError(
                f"Snapshot at step {step} claims to be a delta against step "
                f"{base_step}, which is not earlier than itself. The log is corrupt."
            )

        base = self._materialise(base_step)
        if base is None:
            return None
        out = self.snapshot_dir / f"materialised-{step:06d}.db"
        if not out.exists():
            apply_delta(str(base), path, str(out), verify=True, verbose=False)
        return out

    def mark_rolled_back(self, steps: list[int]) -> int:
        """Flag steps as rolled back. Does not change any weights by itself."""
        placeholders = ",".join("?" * len(steps))
        cur = self.conn.execute(
            f"UPDATE param_updates SET rolled_back = 1 WHERE step IN ({placeholders})",
            steps,
        )
        self.conn.commit()
        return cur.rowcount

    def close(self):
        self.conn.commit()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- reading -----------------------------------------------------------

    def loss_curve(self) -> list[tuple]:
        return self.conn.execute(
            "SELECT step, loss FROM steps WHERE loss IS NOT NULL ORDER BY step"
        ).fetchall()

    def loss_spikes(self, factor: float = 1.5) -> list[tuple]:
        """Steps where loss rose sharply against the previous step."""
        curve = self.loss_curve()
        spikes = []
        for (prev_step, prev_loss), (step, loss) in zip(curve, curve[1:]):
            if prev_loss and loss > prev_loss * factor:
                spikes.append((step, prev_loss, loss))
        return spikes

    def most_changed_params(self, limit: int = 10) -> list[tuple]:
        return self.conn.execute(
            "SELECT param, SUM(grad_norm) AS total, COUNT(*) AS n "
            "FROM param_updates GROUP BY param ORDER BY total DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def step_detail(self, step: int, limit: int = 10) -> list[tuple]:
        return self.conn.execute(
            "SELECT param, grad_norm, grad_max, weight_norm_after FROM param_updates "
            "WHERE step = ? ORDER BY grad_norm DESC LIMIT ?",
            (step, limit),
        ).fetchall()

    def param_history(self, param: str) -> list[tuple]:
        """Every step that touched a parameter, oldest first.

        Returns [(step, grad_norm, grad_mean, grad_max,
                  weight_norm_before, weight_norm_after, rolled_back), ...].
        """
        return self.conn.execute(
            "SELECT step, grad_norm, grad_mean, grad_max, "
            "weight_norm_before, weight_norm_after, rolled_back "
            "FROM param_updates WHERE param = ? ORDER BY step",
            (param,),
        ).fetchall()

    def param_names(self) -> list[str]:
        """All parameter names that appear in the log."""
        return [
            r[0] for r in self.conn.execute(
                "SELECT DISTINCT param FROM param_updates ORDER BY param"
            )
        ]

    def snapshot_steps(self) -> list[int]:
        """Snapshot steps in ascending order."""
        return [
            r[0] for r in self.conn.execute(
                "SELECT step FROM snapshots ORDER BY step"
            )
        ]


def state_dict_to_sqlite(state_dict: dict, db_path: str) -> str:
    """Write a model's parameters into a reminis model database.

    Uses the same schema and shape convention as the format converters, so
    snapshots can be diffed and applied by the existing machinery rather than
    a parallel one.
    """
    from reminis.converter import SCHEMA
    from reminis.dtypes import DTYPE_SYSTEM_SAFETENSORS, SAFETENSORS_DTYPE_IDS

    numpy_to_st = {
        "float64": "F64", "float32": "F32", "float16": "F16",
        "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8",
        "uint8": "U8", "bool": "BOOL",
    }

    Path(db_path).unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)

    for name, tensor in state_dict.items():
        if str(getattr(tensor, "dtype", "")) == "torch.bfloat16":
            # Keep bf16 as bf16 rather than silently promoting to float32: a
            # snapshot that changes dtype is not a snapshot of the model. numpy
            # has no bfloat16, so reinterpret the same bits as int16 to carry
            # them across.
            import torch

            raw = tensor.detach().cpu().contiguous()
            blob = raw.view(torch.int16).numpy().tobytes()
            dtype_name = "BF16"
            shape = list(raw.shape)
            n_elements = int(raw.numel())
        else:
            arr = _to_numpy(tensor)
            arr = np.ascontiguousarray(arr)
            dtype_name = numpy_to_st.get(arr.dtype.name)
            if dtype_name is None:
                raise ValueError(
                    f"Parameter '{name}' has dtype {arr.dtype.name}, which reminis "
                    "cannot store"
                )
            blob = arr.tobytes()
            shape = list(arr.shape)
            n_elements = int(arr.size)

        conn.execute(
            "INSERT OR REPLACE INTO tensors (name, shape, dtype, dtype_id, "
            "n_elements, n_bytes, data) VALUES (?,?,?,?,?,?,?)",
            (name, json.dumps(shape[::-1]), dtype_name,
             SAFETENSORS_DTYPE_IDS[dtype_name], n_elements, len(blob), blob),
        )

    for key, value in (
        ("reminis.source_format", "state_dict"),
        ("reminis.dtype_system", DTYPE_SYSTEM_SAFETENSORS),
    ):
        conn.execute(
            "INSERT OR REPLACE INTO model_meta (key, value, type) VALUES (?,?,'string')",
            (key, value),
        )

    conn.commit()
    conn.close()
    return db_path


def rollback_to_step(
    log: TrainingLog,
    step: int,
    output_db: str,
    verbose: bool = True,
) -> str:
    """Restore the model to its state at a logged snapshot.

    This is a rewind, and it is exact: it reproduces the weights as they were,
    verified against the hash recorded at snapshot time. Everything the run
    learned after that step is discarded along with the bad step.

    It is *not* the same as removing one step from the middle of a trajectory
    and keeping the rest. Every gradient after a bad step was computed from
    the weights that step produced, so those later updates are only valid in
    the context of the step you want gone. Deleting it and replaying them
    yields an approximation, not the model you would have trained without it.
    `reminis` does not pretend otherwise: to genuinely drop a mid-run step you
    have to rewind to before it and train forward again.
    """
    import shutil

    materialised = log._materialise(step)
    if materialised is None:
        available = [
            r[0] for r in log.conn.execute("SELECT step FROM snapshots ORDER BY step")
        ]
        raise ValueError(
            f"No snapshot at step {step}. Snapshots exist at: {available or 'none'}"
        )

    shutil.copyfile(materialised, output_db)

    from reminis.diff import _weights_hash

    conn = sqlite3.connect(output_db)
    actual = _weights_hash(conn)
    conn.close()

    expected = log.conn.execute(
        "SELECT weights_hash FROM snapshots WHERE step = ?", (step,)
    ).fetchone()[0]
    if actual != expected:
        Path(output_db).unlink(missing_ok=True)
        raise ValueError(
            f"Restored weights do not match the hash recorded at step {step}.\n"
            f"  expected: {expected[:16]}...\n  actual:   {actual[:16]}..."
        )

    if verbose:
        print(f"Restored the model as of step {step} -> {output_db}")
        print(f"Verified against the snapshot hash taken during training.")
        print(
            "This is a rewind: updates after this step are gone, not "
            "selectively removed."
        )
    return output_db


def _version() -> str:
    from reminis import __version__

    return __version__
