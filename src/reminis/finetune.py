"""LoRA fine-tuning inside the database.

Everything else in reminis touches finished models: `diff` and `merge` combine
them, `track` logs a training run that happened elsewhere in torch. This adds
the missing verb -- training a LoRA adapter, in-place, with the adapter and
the optimizer state stored as rows in the run database next to the frozen
weights they act on.

Why the shape it has:

  * **LoRA, not full fine-tuning.** A handful of small matrices train; the
    base weights are frozen constants. The output is exactly the structure
    `reminis lora` already converts and `reminis apply` already reconstructs,
    so a run finishes by writing a peft-shaped adapter directory and the
    whole existing delta-pack machinery takes over, hash checks included.
  * **MLX only, for now.** Autodiff is what a backward pass needs, and of the
    three backends only MLX has it (`mx.value_and_grad`). numpy is the
    reference the forward pass is verified against; cupy has no autodiff.
    The run computes in float32 regardless of what the file stores, because
    a gradient is a sum over a whole sequence and float16 saturates at
    65504 -- the Whisper encoder that overflowed in float16 still correlated
    0.956 with the reference, which is what silently-wrong looks like.
  * **Sequences, not batches.** A step is one or more sequences, each run
    through the model as its own teacher-forcing pass and averaged, because
    the forward pass has no batch dimension below `generate()` yet (issue
    #25 tracks the fix). Correctness over throughput while the ground truth
    is still being established.

The run database is a copy of the base model's, with the training tables
added: `steps` and `param_updates` come from `track.LOG_SCHEMA`, so `reminis
log`, `reminis blame` and `reminis bisect` work on a run this module produced
the same way they work on a run torch produced. The adapter itself lives in
`adapter_state` and the Adam moments in `optimizer_state`, both keyed by
step, so a run resumes from the database rather than from a side file that
must match the checkpoint by convention.
"""

import json
import math
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np

from reminis.db import open_for_append
from reminis.kvcache import KVCache
from reminis.model import Model
from reminis.track import TrainingLog

# The training tables, added to a copy of the base model database. `steps`
# and `param_updates` come from track.LOG_SCHEMA, created by TrainingLog.
FINETUNE_SCHEMA = """
CREATE TABLE IF NOT EXISTS finetune_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS adapter_state (
    step   INTEGER NOT NULL,
    tensor TEXT NOT NULL,
    role   TEXT NOT NULL CHECK (role IN ('A','B')),
    dtype  TEXT NOT NULL,
    shape  TEXT NOT NULL,
    data   BLOB NOT NULL,
    PRIMARY KEY (step, tensor, role)
);

CREATE TABLE IF NOT EXISTS optimizer_state (
    step   INTEGER NOT NULL,
    tensor TEXT NOT NULL,
    role   TEXT NOT NULL CHECK (role IN ('A','B')),
    m      BLOB NOT NULL,
    v      BLOB NOT NULL,
    PRIMARY KEY (step, tensor, role)
);

CREATE INDEX IF NOT EXISTS idx_adapter_state_step ON adapter_state(step);
"""

# Target kind -> the segment a tensor name carries for it, in each naming
# convention the database can hold: GGUF's (`blk.5.attn_q.weight`) and
# safetensors' (`model.layers.5.self_attn.q_proj.weight`).
_MODULES = {
    "q": ("attn_q", "q_proj"),
    "k": ("attn_k", "k_proj"),
    "v": ("attn_v", "v_proj"),
    "o": ("attn_output", "o_proj"),
    "gate": ("ffn_gate", "gate_proj"),
    "up": ("ffn_up", "up_proj"),
    "down": ("ffn_down", "down_proj"),
}
ALL_TARGETS = tuple(_MODULES)
DEFAULT_TARGETS = ("q", "k", "v", "o", "gate", "up", "down")

_TARGET_SEGMENTS = {
    segment: kind
    for kind, names in _MODULES.items()
    for segment in names
}


def _kind_of(name: str) -> str | None:
    """Which projection a weight tensor is, in either naming convention."""
    if not name.endswith(".weight"):
        return None
    last = name[: -len(".weight")].split(".")[-1]
    return _TARGET_SEGMENTS.get(last)


def select_targets(conn: sqlite3.Connection, targets) -> list[str]:
    """The base weight tensors the adapter will touch, from either naming.

    Quantized tensors are refused rather than silently broken: a packed
    weight can be multiplied but not differentiated, so training against one
    would produce gradients for the adapter that say nothing true about the
    model. (Training *on top of* frozen quantized weights -- QLoRA -- is the
    natural fit and is not built yet; when it is, this refusal is what it
    replaces.)
    """
    unknown = sorted(set(targets) - set(_MODULES))
    if unknown:
        raise ValueError(
            f"Unknown LoRA targets: {', '.join(unknown)}. "
            f"Expected kinds drawn from {', '.join(_MODULES)}"
        )
    kinds = set(targets)
    rows = conn.execute("SELECT name, dtype FROM tensors").fetchall()
    chosen = []
    refused = []
    for name, dtype in rows:
        kind = _kind_of(name)
        if kind not in kinds:
            continue
        if not dtype.startswith(("F32", "F16", "F64", "BF16")):
            refused.append(f"{name} ({dtype})")
            continue
        chosen.append(name)
    if refused:
        raise ValueError(
            "These targeted tensors are quantized, and a quantized weight "
            "can be multiplied but not differentiated:\n  "
            + "\n  ".join(refused[:5])
            + "\nDequantize the model first, or target only float tensors."
        )
    if not chosen:
        raise ValueError(
            "No base tensor matched the requested LoRA targets "
            f"({', '.join(sorted(kinds))}). This model's tensor names "
            "carry none of the projections looked for."
        )
    return sorted(chosen)


def load_texts(path: str) -> list[str]:
    """Training documents from a JSONL file.

    Each line is either JSON with a `text` field (or `prompt` + `completion`,
    joined with a newline) or, failing that, plain text.
    """
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                texts.append(line)
                continue
            if isinstance(row, str):
                texts.append(row)
                continue
            if "text" in row:
                texts.append(str(row["text"]))
            elif "prompt" in row or "completion" in row:
                prompt = str(row.get("prompt", ""))
                completion = str(row.get("completion", ""))
                texts.append(f"{prompt}\n{completion}".strip())
    if not texts:
        raise ValueError(f"{path} holds no usable training documents")
    return texts


def _np_to_st_dtype(name: str) -> str:
    mapping = {
        "float64": "F64", "float32": "F32", "float16": "F16",
        "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8",
        "uint8": "U8", "bool": "BOOL",
    }
    out = mapping.get(name)
    if out is None:
        raise ValueError(f"numpy dtype {name} has no safetensors equivalent")
    return out


def write_safetensors(tensors: dict, path: str) -> str:
    """Write ``{name: np.ndarray}`` as one safetensors file.

    A small direct writer, because the adapter has no database to travel
    through: it is two small matrices per module and a config file, and the
    shape it is written in is peft's, so `reminis lora` -- and peft itself --
    can read it back.
    """
    import reminis

    header = {}
    offset = 0
    ordered = []
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        n_bytes = arr.nbytes
        header[name] = {
            "dtype": _np_to_st_dtype(arr.dtype.name),
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + n_bytes],
        }
        ordered.append(arr)
        offset += n_bytes
    header["__metadata__"] = {
        "format": "pt", "reminis_version": reminis.__version__,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    with open(path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        for arr in ordered:
            f.write(arr.tobytes())
    return path


def _scaling(rank: int, alpha: float, use_rslora: bool) -> float:
    """peft's convention: alpha/r, or alpha/sqrt(r) for rank-stabilised LoRA."""
    if use_rslora:
        return float(alpha) / float(math.sqrt(rank))
    return float(alpha) / float(rank)


def read_run_meta(path: str) -> dict:
    """The `finetune_meta` rows of a run database, checked to be one."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Run database not found: {path}")
    conn = sqlite3.connect(str(path))
    try:
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "finetune_meta" not in tables:
            raise ValueError(
                f"{path} is not a fine-tuning run database (no finetune_meta "
                "table). Start one with `reminis finetune train`."
            )
        meta = dict(conn.execute("SELECT key, value FROM finetune_meta"))
        if meta.get("kind") != "lora":
            raise ValueError(
                f"{path} records kind={meta.get('kind')!r}, not 'lora'."
            )
        return meta
    finally:
        conn.close()


def save_adapter(conn: sqlite3.Connection, *, rank: int, alpha: float,
                 use_rslora: bool, targets, step: int, out_dir: str,
                 quiet: bool = False) -> str:
    """Write a run's adapter at `step` as a peft-shaped directory.

    `adapter_model.safetensors` with `...lora_A.weight` / `...lora_B.weight`
    names and `adapter_config.json` -- the form `reminis lora` reads and
    peft itself loads. float32 throughout.

    Tensor names carry the run database's own names, stripped to the module
    path (`blk.0.attn_q.lora_A.weight`). That is what the delta-pack
    converter matches on, and it is peft-loadable too when the model's
    module paths use those same names -- which they do when the adapter is
    loaded by tools built on the GGUF naming, and not by transformers'
    class-per-architecture models, whose modules are spelled `q_proj` and
    friends. Mapping between the two spellings needs the base model's
    module tree, which this export deliberately does not guess at: a wrong
    guess would load quietly into peft with every adapter at its
    initialization instead.
    """
    rows = conn.execute(
        "SELECT tensor, role, shape, data FROM adapter_state WHERE step = ?",
        (step,),
    ).fetchall()
    if not rows:
        raise ValueError(f"No adapter checkpoint at step {step}")

    tensors = {}
    for tensor, role, shape_json, blob in rows:
        module = (tensor[: -len(".weight")]
                  if tensor.endswith(".weight") else tensor)
        arr = np.frombuffer(blob, dtype=np.float32).reshape(
            json.loads(shape_json)).copy()
        tensors[f"{module}.lora_{role}.weight"] = arr

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_safetensors(tensors, str(out_dir / "adapter_model.safetensors"))
    kinds = sorted({k for n in targets if (k := _kind_of(n))})
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": alpha,
        "use_rslora": use_rslora,
        "target_modules": [_MODULES[k][1] for k in kinds],
        "reminis_step": step,
    }, indent=2) + "\n")
    if not quiet:
        print(f"Adapter at step {step} -> {out_dir}")
    return str(out_dir)


class LoRATrainer:
    """Train a LoRA adapter against a frozen base model, in the database.

    A fresh run copies the base database to `out_db` and adds the training
    tables; `resume=True` reopens a run database instead, restoring the
    adapter and the Adam moments from its newest checkpoint.
    """

    def __init__(self, base_db: str, out_db: str, *, rank: int = 8,
                 alpha: float = 16.0, targets=DEFAULT_TARGETS,
                 seed: int = 0, use_rslora: bool = False,
                 run_name: str | None = None, verbose: bool = True,
                 resume: bool = False):
        self.verbose = verbose
        self.backend = self._make_backend()
        self.mx = self.backend.mx

        if resume:
            self._open_existing(out_db)
            return

        if rank < 1:
            raise ValueError("LoRA rank must be at least 1")
        self.base_db = str(Path(base_db))
        self.out_db = str(Path(out_db))
        if not Path(self.base_db).exists():
            raise FileNotFoundError(
                f"Base model database not found: {self.base_db}"
            )
        if Path(self.out_db).resolve() == Path(self.base_db).resolve():
            raise ValueError(
                "--out must differ from the base database: the run writes "
                "tables into its own copy, and training in place would leave "
                "no untouched base to diff and apply against."
            )

        # The run database *is* the model database: a copy of the base, with
        # the training tables beside the frozen weights. Everything the run
        # needs to resume -- adapter, Adam moments, loss history -- lives in
        # this one file.
        Path(self.out_db).unlink(missing_ok=True)
        shutil.copyfile(self.base_db, self.out_db)
        self.conn = open_for_append(self.out_db)
        self.conn.executescript(FINETUNE_SCHEMA)

        self.model = Model(self.out_db, backend=self.backend)
        cfg = self.model.cfg
        if cfg.is_moe:
            raise ValueError(
                "Mixture-of-experts models are refused: the router's top-k "
                "is a non-differentiable gather, so training against it "
                "would produce adapter gradients that do not mean what they "
                "appear to. Not built yet, and stated rather than shipped "
                "half-working."
            )
        from reminis.arch import Arch
        if type(cfg.spec).block is not Arch.block:
            raise ValueError(
                f"Architecture '{type(cfg.spec).__name__}' has its own block "
                "computation, which this training path does not cover yet. "
                "Dense attention models are supported."
            )

        self.rank = rank
        self.alpha = float(alpha)
        self.use_rslora = use_rslora
        self.lora_scale = _scaling(rank, alpha, use_rslora)
        self.targets = select_targets(self.conn, targets)

        self._init_meta(run_name)
        self._init_params(seed)

        self.log = TrainingLog(self.out_db, run_name=run_name)
        self.log.set_meta(
            finetune_kind="lora",
            lora_rank=str(rank),
            lora_alpha=repr(alpha),
            lora_scaling=repr(self.lora_scale),
            lora_targets=",".join(sorted(set(targets))),
        )

        self._step = 0
        self._adam_t = 0
        self._moments = self._zero_moments()
        self._apply_params(self.params)

    # -- construction helpers ---------------------------------------------

    def _make_backend(self):
        """MLX at float32, which is what a backward pass needs."""
        from reminis.backend import MLXBackend

        if not MLXBackend.available():
            raise ValueError(
                "Fine-tuning needs the mlx backend (the only one of the three "
                "with autodiff), and no Metal device is available on this "
                "machine."
            )
        import mlx.core as mx

        return MLXBackend(compute_dtype=mx.float32)

    def _open_existing(self, out_db: str):
        """Resume: reopen a run database and restore its newest checkpoint."""
        path = Path(out_db)
        if not path.exists():
            raise FileNotFoundError(f"Run database not found: {out_db}")
        meta = read_run_meta(out_db)
        self.base_db = meta.get("base_db", str(path))
        self.out_db = str(path)
        self.rank = int(meta.get("rank", 8))
        self.alpha = float(meta.get("alpha", "16.0"))
        self.use_rslora = meta.get("use_rslora") == "true"
        self.lora_scale = _scaling(self.rank, self.alpha, self.use_rslora)

        self.conn = open_for_append(self.out_db)
        self.conn.executescript(FINETUNE_SCHEMA)
        self.model = Model(self.out_db, backend=self.backend)
        # finetune_meta stores full tensor names; kinds are what
        # select_targets wants, so re-derive them from the names.
        stored = [t for t in meta.get("targets", "").split(",") if t]
        kinds = []
        for name in stored:
            kind = _kind_of(name)
            if kind and kind not in kinds:
                kinds.append(kind)
        self.targets = select_targets(self.conn, kinds)

        self.log = TrainingLog(self.out_db)

        self._step = 0
        self._adam_t = 0
        self._init_params(seed=0)  # overwritten row by row below
        self._moments = self._zero_moments()
        last = self.conn.execute(
            "SELECT MAX(step) FROM adapter_state").fetchone()[0]
        if last is not None:
            self._load_checkpoint(int(last))
        elif self.verbose:
            print("No checkpoints in this run database yet; starting from "
                  "a fresh adapter.")
        self._apply_params(self.params)

    def _init_meta(self, run_name):
        self.conn.executemany(
            "INSERT OR REPLACE INTO finetune_meta (key, value) VALUES (?,?)",
            [
                ("kind", "lora"),
                ("base_db", self.base_db),
                ("base_file", Path(self.base_db).name),
                ("rank", str(self.rank)),
                ("alpha", repr(self.alpha)),
                ("scaling", repr(self.lora_scale)),
                ("use_rslora", "true" if self.use_rslora else "false"),
                ("targets", ",".join(self.targets)),
                ("created_at", repr(time.time())),
            ],
        )
        self.conn.commit()

    def _init_params(self, seed: int):
        """A: Kaiming-uniform like peft; B: zeros.

        B starting at zero is the convention because it makes the merged
        model equal the base at step 0 -- the run begins where the base is.
        The first updates all land on B, and A starts learning once B is
        nonzero. That is peft's own behaviour, so an adapter trained here is
        continuous with one trained there.
        """
        rng = np.random.default_rng(seed)
        self.params = {}
        self.shapes: dict[str, tuple[int, int]] = {}
        for name in self.targets:
            shape = json.loads(self.conn.execute(
                "SELECT shape FROM tensors WHERE name = ?", (name,)
            ).fetchone()[0])
            # Stored reversed (GGUF convention): the row's [in, out] is the
            # logical (out, in).
            out_features, in_features = shape[1], shape[0]
            self.shapes[name] = (in_features, out_features)
            if not self._has_checkpoint(name):
                bound = 1.0 / math.sqrt(in_features)
                a = rng.uniform(-bound, bound,
                                size=(self.rank, in_features)).astype(np.float32)
                b = np.zeros((out_features, self.rank), dtype=np.float32)
                self.params[name] = {
                    "A": self.backend.from_numpy(a),
                    "B": self.backend.from_numpy(b),
                }
        for name in self.targets:
            self.params.setdefault(name, {
                "A": self.backend.zeros((self.rank, self.shapes[name][0])),
                "B": self.backend.zeros((self.shapes[name][1], self.rank)),
            })

    def _has_checkpoint(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM adapter_state WHERE tensor = ?", (name,)
        ).fetchone()
        return bool(row and row[0])

    def _zero_moments(self) -> dict:
        return {
            name: {
                role: (self.backend.zeros(tuple(arr.shape)),
                       self.backend.zeros(tuple(arr.shape)))
                for role, arr in pairs.items()
            }
            for name, pairs in self.params.items()
        }

    # -- the forward pass and its loss -------------------------------------

    def _apply_params(self, p: dict) -> None:
        self.model.lora = {
            name: (pair["A"], pair["B"]) for name, pair in p.items()
        }
        self.model.lora_scale = self.lora_scale

    def _ce(self, logits, token_ids):
        """Mean cross-entropy over the shifted positions of one sequence."""
        mx = self.mx
        labels = mx.array(np.asarray(token_ids[1:], dtype=np.int32))
        shifted = logits[:-1]
        lse = mx.logsumexp(shifted, axis=-1)
        picked = mx.take_along_axis(
            shifted, labels.reshape(-1, 1), axis=-1
        ).reshape(-1)
        return mx.mean(lse - picked)

    def _batch_loss(self, batch, p: dict):
        """Mean loss over the batch's sequences, as one autograd graph.

        Each sequence is its own teacher-forcing pass: `all_positions=True`
        keeps logits at every position, and a cache of exactly the
        sequence's length is thrown away afterwards. The forward pass has no
        batch dimension below `generate()` yet, so this is a loop rather
        than a batched matmul -- correct, and honest about not being fast.
        """
        self._apply_params(p)
        model = self.model
        losses = []
        for seq in batch:
            tokens = np.asarray(seq, dtype=np.int32)
            cache = KVCache(model.cfg.n_layers, capacity=len(tokens),
                            backend=model.backend)
            logits = model.forward(tokens.tolist(), cache, 0,
                                   all_positions=True, grad=True)
            losses.append(self._ce(logits, tokens))
        mx = self.mx
        if len(losses) == 1:
            return losses[0]
        return mx.mean(mx.stack(losses))

    # -- optimizer ---------------------------------------------------------

    def _adamw(self, grads: dict, lr: float, beta1: float, beta2: float,
               eps: float, weight_decay: float) -> None:
        mx = self.mx
        t = self._adam_t = self._adam_t + 1
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        for name, pairs in self.params.items():
            for role in ("A", "B"):
                g = grads[name][role]
                m, v = self._moments[name][role]
                m = beta1 * m + (1.0 - beta1) * g
                v = beta2 * v + (1.0 - beta2) * g * g
                m_hat = m / bc1
                v_hat = v / bc2
                p = pairs[role]
                pairs[role] = p - lr * (
                    m_hat / (mx.sqrt(v_hat) + eps) + weight_decay * p
                )
                self._moments[name][role] = (m, v)
        mx.eval(*(a for pairs in self.params.values() for a in pairs.values()))
        mx.eval(*(x for name in self._moments
                  for x in self._moments[name].values()))

    def _clip_grads(self, grads: dict, max_grad_norm: float):
        """Global-norm clipping, as HuggingFace's default 1.0 does.

        Returns the pre-clip norm when clipping fired, None when it did not.
        """
        if not max_grad_norm or max_grad_norm <= 0:
            return None
        mx = self.mx
        sq = None
        for pairs in grads.values():
            for g in pairs.values():
                part = mx.sum(g * g)
                sq = part if sq is None else sq + part
        norm = float(np.asarray(mx.sqrt(sq)))
        if norm <= max_grad_norm:
            return None
        scale = max_grad_norm / norm
        for pairs in grads.values():
            for role, g in pairs.items():
                pairs[role] = g * scale
        return norm

    # -- persistence -------------------------------------------------------

    def checkpoint(self, step: int) -> None:
        """Write the adapter and the Adam moments as rows at this step.

        The adapter is stored in float32 whatever it computes in, which is
        what `adapter_state` and the pack format both expect. Both tables are
        keyed by step, so a run is a timeline: any earlier step's optimizer
        state is still there to resume from.
        """
        for name, pairs in self.params.items():
            for role, arr in pairs.items():
                blob = self.backend.to_bytes(arr, "F32")
                self.conn.execute(
                    "INSERT OR REPLACE INTO adapter_state "
                    "(step, tensor, role, dtype, shape, data) VALUES (?,?,?,?,?,?)",
                    (step, name, role, "F32",
                     json.dumps(list(arr.shape)), blob),
                )
        for name in self._moments:
            for role in ("A", "B"):
                m, v = self._moments[name][role]
                self.conn.execute(
                    "INSERT OR REPLACE INTO optimizer_state "
                    "(step, tensor, role, m, v) VALUES (?,?,?,?,?)",
                    (step, name, role,
                     self.backend.to_bytes(m, "F32"),
                     self.backend.to_bytes(v, "F32")),
                )
        self.conn.execute(
            "INSERT OR REPLACE INTO finetune_meta (key, value) VALUES (?,?)",
            ("last_step", str(step)),
        )
        self.conn.commit()

    def _load_checkpoint(self, step: int) -> None:
        """Restore adapter and Adam moments from the rows at `step`.

        Mismatches are rejected loudly rather than applied half-ignored: a
        resumed run that disagrees with its own checkpoint is corrupt, and
        the error should name the step.
        """
        rows = self.conn.execute(
            "SELECT tensor, role, shape, data FROM adapter_state WHERE step = ?",
            (step,),
        ).fetchall()
        if not rows:
            raise ValueError(f"No adapter checkpoint at step {step}")
        loaded = {}
        for tensor, role, shape_json, blob in rows:
            arr = np.frombuffer(blob, dtype=np.float32).reshape(
                json.loads(shape_json)).copy()
            loaded.setdefault(tensor, {})[role] = arr
        missing = set(self.params) - set(loaded)
        if missing:
            raise ValueError(
                f"Checkpoint at step {step} is missing factors for: "
                f"{', '.join(sorted(missing)[:5])}. The run's targets were "
                "probably changed between the checkpoint and now."
            )
        for name in self.params:
            for role in ("A", "B"):
                arr = loaded[name][role]
                shape = tuple(self.params[name][role].shape)
                if arr.shape != shape:
                    raise ValueError(
                        f"Checkpoint at step {step} has {name}.{role} shaped "
                        f"{arr.shape}, but the run's targets define {shape}."
                    )
                self.params[name][role] = self.backend.from_numpy(arr)
        self._step = step
        opt_rows = self.conn.execute(
            "SELECT tensor, role, m, v FROM optimizer_state WHERE step = ?",
            (step,),
        ).fetchall()
        if opt_rows:
            for tensor, role, m_blob, v_blob in opt_rows:
                if tensor not in self.params:
                    continue
                shape = self.params[tensor][role].shape
                m = np.frombuffer(m_blob, dtype=np.float32).reshape(shape)
                v = np.frombuffer(v_blob, dtype=np.float32).reshape(shape)
                self._moments[tensor][role] = (
                    self.backend.from_numpy(m.copy()),
                    self.backend.from_numpy(v.copy()),
                )
            self._adam_t = step

    # -- the training loop -------------------------------------------------

    def train(self, data_path: str, *, steps: int, lr: float = 1e-4,
              batch_size: int = 1, seq_len: int = 256,
              checkpoint_every: int = 20, log_every: int = 1,
              weight_decay: float = 0.0, max_grad_norm: float = 1.0,
              beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
              quiet: bool = False) -> dict:
        """Run the loop.

        Returns a summary dict. A trainer opened with `resume=True`
        continues from its newest checkpoint: adapter, Adam moments and step
        counter all come back from the database, and `steps` is how many
        *more* to run.
        """
        import mlx.core as mx

        if batch_size < 1 or seq_len < 2:
            raise ValueError("batch_size >= 1 and seq_len >= 2 are required")

        if self._step == 0 and self.conn.execute(
            "SELECT COUNT(*) FROM adapter_state"
        ).fetchone()[0]:
            raise ValueError(
                f"{self.out_db} already holds training checkpoints. Open it "
                "with resume=True (or `reminis finetune train --resume`) to "
                "continue it, or point --out somewhere new to start over."
            )

        texts = load_texts(data_path)
        sequences = []
        skipped = 0
        for text in texts:
            ids = self.model.tokenizer.encode(text)
            if len(ids) < seq_len:
                skipped += 1
                continue
            sequences.append(np.asarray(ids[:seq_len], dtype=np.int32))
        if not sequences:
            raise ValueError(
                f"Every one of the {len(texts)} documents is shorter than "
                f"seq_len={seq_len}. Lower --seq-len or use longer data."
            )

        rng = np.random.default_rng(0)
        order = list(range(len(sequences)))
        rng.shuffle(order)
        cursor = 0
        epochs_started = 0

        self._apply_params(self.params)

        t0 = time.time()
        started = self._step
        losses = []
        for i in range(steps):
            batch = []
            for _ in range(batch_size):
                if cursor >= len(order):
                    epochs_started += 1
                    rng.shuffle(order)
                    cursor = 0
                batch.append(sequences[order[cursor]])
                cursor += 1

            step_number = self._step + 1

            def loss_fn(p, batch=batch):
                return self._batch_loss(batch, p)

            loss, grads = mx.value_and_grad(loss_fn)(self.params)

            clipped = self._clip_grads(grads, max_grad_norm)
            self._adamw(grads, lr, beta1, beta2, eps, weight_decay)

            loss_value = float(np.asarray(loss))
            losses.append(loss_value)

            named_grads = {}
            named_weights = {}
            for name in self.params:
                for role in ("A", "B"):
                    full = f"{name}.lora_{role}"
                    named_grads[full] = self.backend.to_numpy(grads[name][role])
                    named_weights[full] = self.backend.to_numpy(
                        self.params[name][role])
            epoch = epochs_started + cursor / max(len(order), 1)
            self.log.log_step(
                step_number, named_grads, named_weights,
                loss=loss_value, epoch=epoch, learning_rate=lr,
            )
            self._step = step_number

            if not quiet and (i % log_every == 0 or i == steps - 1):
                clip_note = (f"  grad norm {clipped:.3f} (clipped)"
                             if clipped is not None else "")
                print(f"  step {step_number:>5}  loss {loss_value:.4f}"
                      f"{clip_note}")

            if step_number % checkpoint_every == 0 or i == steps - 1:
                self.checkpoint(step_number)

        elapsed = time.time() - t0
        summary = {
            "steps": self._step - started,
            "total_step": self._step,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_min": min(losses),
            "sequences": len(sequences),
            "skipped_short": skipped,
            "targets": len(self.targets),
            "rank": self.rank,
            "lora_params": sum(
                arr.size for pairs in self.params.values()
                for arr in pairs.values()
            ),
            "elapsed": elapsed,
            "out_db": self.out_db,
        }
        if not quiet:
            _print_train_report(summary)
        return summary

    def save_adapter(self, out_dir: str, step: int | None = None,
                     quiet: bool = False) -> str:
        step = step if step is not None else self._step
        return save_adapter(
            self.conn, rank=self.rank, alpha=self.alpha,
            use_rslora=self.use_rslora, targets=self.targets,
            step=step, out_dir=out_dir, quiet=quiet,
        )

    def close(self):
        self.log.close()
        self.conn.close()
        self.model.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def merge_run(run_db: str, output_db: str, *, base_db: str | None = None,
              step: int | None = None, keep_adapter: str | None = None,
              quiet: bool = False) -> str:
    """Fold a finished run's adapter into a full model database.

    The adapter is exported in peft's own format, converted to a delta pack
    by `reminis lora` -- which records the base's fingerprint and refuses a
    wrong base -- and applied by `reminis apply`, which verifies the result
    against a hash recorded when the pack was written. Nothing here trusts
    the arithmetic on its own.
    """
    from reminis.diff import apply_delta
    from reminis.lora import lora_to_delta_pack

    meta = read_run_meta(run_db)
    last = sqlite3.connect(run_db).execute(
        "SELECT MAX(step) FROM adapter_state").fetchone()[0]
    sqlite3.connect(run_db).close()
    step = step if step is not None else (int(last) if last is not None else 0)
    base = base_db or meta.get("base_db") or run_db
    if not Path(base).exists():
        raise FileNotFoundError(
            f"The run's base database is recorded as {base}, which does not "
            "exist. Pass --base with the original model database."
        )

    with tempfile.TemporaryDirectory(prefix="reminis-merge-") as tmp:
        conn = sqlite3.connect(run_db)
        try:
            adapter_dir = Path(tmp) / "adapter"
            save_adapter(conn, rank=int(meta.get("rank", 8)),
                         alpha=float(meta.get("alpha", "16.0")),
                         use_rslora=meta.get("use_rslora") == "true",
                         targets=meta.get("targets", "").split(","),
                         step=step, out_dir=str(adapter_dir), quiet=True)
        finally:
            conn.close()
        pack_path = Path(tmp) / "pack.db"
        lora_to_delta_pack(str(adapter_dir), base, str(pack_path),
                           verbose=not quiet)
        apply_delta(base, str(pack_path), output_db,
                    verify=True, verbose=not quiet)

    if keep_adapter:
        conn = sqlite3.connect(run_db)
        try:
            save_adapter(conn, rank=int(meta.get("rank", 8)),
                         alpha=float(meta.get("alpha", "16.0")),
                         use_rslora=meta.get("use_rslora") == "true",
                         targets=meta.get("targets", "").split(","),
                         step=step, out_dir=keep_adapter, quiet=quiet)
        finally:
            conn.close()

    if not quiet:
        print(f"\nMerged model: {output_db}  (base {Path(base).name} "
              f"+ adapter at step {step})")
    return output_db


def _print_train_report(s: dict) -> None:
    print(f"\n  {s['steps']} steps (total {s['total_step']}), "
          f"loss {s['loss_first']:.4f} -> {s['loss_last']:.4f} "
          f"(best {s['loss_min']:.4f})")
    print(f"  adapter: {s['targets']} tensors, rank {s['rank']}, "
          f"{s['lora_params']:,} trainable parameters")
    print(f"  data: {s['sequences']} sequences used, "
          f"{s['skipped_short']} shorter than seq_len")
    rate = s["steps"] / s["elapsed"] if s["elapsed"] else 0.0
    print(f"  {s['elapsed']:.1f}s, {rate:.2f} steps/s")
    print(f"  run database: {s['out_db']}")
    print("\n  Next: `reminis finetune merge <run.db> -o merged.db` to fold "
          "the adapter\n  into a full model, or `reminis log <run.db>` to "
          "inspect the run.")
