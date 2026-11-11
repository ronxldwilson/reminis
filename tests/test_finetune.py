"""LoRA fine-tuning inside the run database.

The core question this file answers: does `LoRATrainer` compute the same
gradients peft does? Everything else -- the tables, the checkpoints, the
merge -- is plumbing around that one arithmetic fact. The reference is peft
on the same tiny Qwen2 with the same adapter values injected: one backward
pass, then the gradients compared. Equality is the bar, since both are
float32 on the same machine over the same sequence; only operation order
may differ, and with these shapes it doesn't.

The merge needs no arithmetic check of its own: `merge_run` goes through
`reminis lora` and `reminis apply`, whose hash verification refuses a wrong
result -- but this file still checks the merged weights directly, because a
hash of a consistently-wrong pipeline would pass.
"""

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reminis.kvcache import KVCache  # noqa: E402
from reminis.safetensors_io import safetensors_to_sqlite  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
mx = pytest.importorskip("mlx.core", reason="fine-tuning needs MLX autodiff")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from reminis.backend import MLXBackend  # noqa: E402
from reminis.dtypes import to_float32  # noqa: E402
from reminis.finetune import (  # noqa: E402
    ALL_TARGETS,
    LoRATrainer,
    merge_run,
    read_run_meta,
    save_adapter,
    select_targets,
)
from reminis.model import Model  # noqa: E402

VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, FFN = 64, 32, 2, 4, 2, 64
RANK, ALPHA = 4, 8.0
# (block, peft module, gguf segment, out_features)
MODULES = [("q_proj", "attn_q", HIDDEN),
           ("v_proj", "attn_v", KV_HEADS * (HIDDEN // HEADS))]

TMP = Path(__file__).parent / "tmp_test_finetune"


# -- fixtures -------------------------------------------------------------

def build_model(directory: Path) -> None:
    """The tiny Qwen2 of test_safetensors_run, with a tokenizer it can run."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=FFN,
                      num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                      num_key_value_heads=KV_HEADS, rms_norm_eps=1e-06,
                      rope_theta=1000000.0, max_position_embeddings=128)
    Qwen2ForCausalLM(cfg).float().save_pretrained(directory)

    vocab = {chr(ord("a") + i): i for i in range(26)}
    vocab.update({str(i): 26 + i for i in range(10)})
    vocab["Ġ"] = 36
    merges = ["a b", "c d"]
    for pair in merges:
        joined = pair.replace(" ", "")
        if joined not in vocab:
            vocab[joined] = len(vocab)
    while len(vocab) < VOCAB:
        vocab[f"<extra_{len(vocab)}>"] = len(vocab)
    (directory / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
        "added_tokens": [],
    }))


@pytest.fixture(scope="module")
def base_db(tmp):
    db = safetensors_to_sqlite(str(TMP / "tiny"), str(TMP / "tiny.db"),
                               verbose=False)
    return db


@pytest.fixture(scope="module", autouse=True)
def _model(tmp):
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    build_model(TMP / "tiny")
    yield
    shutil.rmtree(TMP, ignore_errors=True)


@pytest.fixture(scope="module")
def data(tmp):
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(60)]
    path = TMP / "data.jsonl"
    with open(path, "w") as f:
        for _ in range(40):
            f.write(json.dumps({"text": " ".join(rng.choice(words, size=30))}) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def seq():
    """One fixed 16-token sequence the gradient comparison runs on."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, VOCAB, size=16).astype(np.int32)


def _open_trainer(base_db):
    """A trainer shell with no database: only the forward/backward math."""
    trainer = LoRATrainer.__new__(LoRATrainer)
    trainer.backend = MLXBackend(compute_dtype=mx.float32)
    trainer.mx = mx
    trainer.model = Model(base_db, backend=trainer.backend)
    return trainer


def _set_factors(model, factors):
    """Inject {(block, hf_segment_or_gguf): (A, B)} into a Model's lora dict."""
    model.lora_scale = ALPHA / RANK
    model.lora = {}
    for block in range(LAYERS):
        for hf, gguf, _ in MODULES:
            a, b = factors[(block, hf)]
            model.lora[f"blk.{block}.{gguf}.weight"] = (mx.array(a), mx.array(b))


# -- unit checks ----------------------------------------------------------

def test_select_targets(base_db):
    """The kinds present in this model each pick their tensors out."""
    conn = sqlite3.connect(base_db)
    try:
        got = select_targets(conn, list(ALL_TARGETS))
        assert [t for t in got if "attn_q" in t] == [
            f"blk.{i}.attn_q.weight" for i in range(LAYERS)]
        # Every weight tensor of a targeted kind was chosen; biases never are.
        assert not any(n.endswith(".bias") for n in got)
        with pytest.raises(ValueError, match="Unknown LoRA targets"):
            select_targets(conn, ["nope"])
    finally:
        conn.close()


def test_fresh_run_refuses_in_place():
    """Training in place would leave no untouched base to diff against."""
    with pytest.raises(ValueError, match="--out must differ"):
        LoRATrainer(TMP / "tiny.db", TMP / "tiny.db")


def test_gradients_match_peft(base_db, seq, tmp):
    """One backward pass, reminis vs peft, on identical adapter values."""
    from peft import LoraConfig, get_peft_model

    # The trainer's own init: A Kaiming-uniform at 1/sqrt(in), B zeros.
    # q and v both take the hidden state in; v's output is the kv width.
    rng = np.random.default_rng(0)
    factors = {}
    for block in range(LAYERS):
        for hf, _, out_features in MODULES:
            bound = 1.0 / np.sqrt(HIDDEN)
            a = rng.uniform(-bound, bound, size=(RANK, HIDDEN)).astype(np.float32)
            b = np.zeros((out_features, RANK), dtype=np.float32)
            factors[(block, hf)] = (a, b)

    trainer = _open_trainer(base_db)
    try:
        _set_factors(trainer.model, factors)
        trainer.lora_scale = ALPHA / RANK
        # _batch_loss differentiates over a {name: {"A": ..., "B": ...}} dict,
        # which is the shape the optimizer's state is kept in too.
        params = {name: {"A": pair[0], "B": pair[1]}
                  for name, pair in trainer.model.lora.items()}
        loss_m, grads = mx.value_and_grad(
            lambda p: trainer._batch_loss([seq], p))(params)
        mx.eval(loss_m)
    finally:
        trainer.model.close()

    # peft on the same base weights (torch.manual_seed(7) rebuilds them) and
    # the same factors.
    torch.manual_seed(7)
    cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=FFN,
                      num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                      num_key_value_heads=KV_HEADS, rms_norm_eps=1e-06,
                      rope_theta=1000000.0, max_position_embeddings=128)
    pm = get_peft_model(Qwen2ForCausalLM(cfg).float(),
                        LoraConfig(r=RANK, lora_alpha=ALPHA,
                                   target_modules=[hf for hf, _, _ in MODULES]))
    with torch.no_grad():
        for n, mod in pm.named_modules():
            if hasattr(mod, "lora_A") and "default" in n:
                block = int(n.split("layers.")[1].split(".")[0])
                hf = "q_proj" if "q_proj" in n else "v_proj"
                a, b = factors[(block, hf)]
                mod.lora_A["default"].weight.copy_(torch.from_numpy(a))
                mod.lora_B["default"].weight.copy_(torch.from_numpy(b))

    seq_t = torch.from_numpy(seq.astype(np.int64)).unsqueeze(0)
    pm.train()
    logits = pm(seq_t).logits[0].float()
    loss_p = torch.nn.functional.cross_entropy(logits[:-1], seq_t[0, 1:])
    loss_p.backward()

    ref = {}
    for n, mod in pm.named_modules():
        if hasattr(mod, "lora_A") and "default" in n:
            block = int(n.split("layers.")[1].split(".")[0])
            hf = "q_proj" if "q_proj" in n else "v_proj"
            ref[(block, hf)] = (mod.lora_A["default"].weight.grad.numpy(),
                                mod.lora_B["default"].weight.grad.numpy())

    assert abs(float(loss_m) - float(loss_p)) < 1e-4, (float(loss_m), float(loss_p))
    worst = 0.0
    for key, (ga_p, gb_p) in ref.items():
        g = grads[f"blk.{key[0]}.{'attn_q' if key[1] == 'q_proj' else 'attn_v'}.weight"]
        scale = max(np.abs(ga_p).max(), np.abs(gb_p).max(), 1e-30)
        worst = max(worst,
                    np.abs(np.asarray(g["A"]) - ga_p).max() / scale,
                    np.abs(np.asarray(g["B"]) - gb_p).max() / scale)
    assert worst < 1e-5, f"gradient mismatch vs peft: {worst:.3e}"


# -- the whole run --------------------------------------------------------

def test_train_and_resume(base_db, data):
    """Train, checkpoint, stop, resume, and land on the same total step."""
    run = str(TMP / "run.db")
    trainer = LoRATrainer(base_db, run, rank=RANK, alpha=ALPHA, seed=0)
    try:
        s = trainer.train(data, steps=5, lr=1e-3, seq_len=16, checkpoint_every=2,
                          batch_size=2, quiet=True)
        assert s["steps"] == 5 and s["total_step"] == 5
        assert s["loss_last"] < s["loss_first"], "loss did not decrease"

        meta = read_run_meta(run)
        assert meta["rank"] == str(RANK)
        assert float(meta["alpha"]) == ALPHA
        assert meta["kind"] == "lora"
        assert "attn_q" in meta["targets"]

        conn = sqlite3.connect(run)
        steps = [r[0] for r in conn.execute(
            "SELECT DISTINCT step FROM adapter_state ORDER BY step")]
        conn.close()
        assert steps == [2, 4, 5], f"checkpoints at {steps}"
    finally:
        trainer.close()

    trainer = LoRATrainer(base_db, run, resume=True)
    try:
        assert trainer._step == 5
        s = trainer.train(data, steps=3, lr=1e-3, seq_len=16, checkpoint_every=2,
                          batch_size=2, quiet=True)
        assert s["total_step"] == 8
    finally:
        trainer.close()

    with pytest.raises(FileNotFoundError):
        LoRATrainer(base_db, str(TMP / "missing.db"), resume=True)


def test_merge_run(base_db, data):
    """Merged weights are exactly base + scaling * B @ A, and the model runs."""
    run = str(TMP / "run.db")
    merged = merge_run(run, str(TMP / "merged.db"), base_db=base_db, quiet=True)

    conn = sqlite3.connect(run)
    rows = conn.execute(
        "SELECT tensor, role, shape, data FROM adapter_state WHERE step = 8"
    ).fetchall()
    conn.close()
    factors = {}
    for tensor, role, shape_json, blob in rows:
        arr = np.frombuffer(blob, dtype=np.float32).reshape(json.loads(shape_json))
        factors.setdefault(tensor, {})[role] = arr

    worst = 0.0
    for name, pair in factors.items():
        conn = sqlite3.connect(base_db)
        shape_json, dtype, blob = conn.execute(
            "SELECT shape, dtype, data FROM tensors WHERE name=?",
            (name,)).fetchone()
        conn.close()
        st_shape = json.loads(shape_json)
        # The blob's row-major layout is (stored[1], stored[0]) -- (out, in) --
        # and the pack folds the scaling into B, so the merge is a plain matmul.
        m, n = st_shape[1], st_shape[0]
        w = to_float32(blob, dtype).reshape(m, n)
        conn = sqlite3.connect(merged)
        mshape, mdtype, mblob = conn.execute(
            "SELECT shape, dtype, data FROM tensors WHERE name=?",
            (name,)).fetchone()
        conn.close()
        w_m = to_float32(mblob, mdtype).reshape(m, n)
        expected = w + (pair["B"] @ pair["A"]) * (ALPHA / RANK)
        worst = max(worst,
                    np.abs(w_m - expected).max() / max(np.abs(expected).max(), 1e-30))
    assert worst < 1e-6, f"merged weights off base + B@A: {worst:.3e}"

    model = Model(merged)
    try:
        logits = model.forward([1, 2, 3, 4], KVCache(model.cfg.n_layers), 0)
        assert np.isfinite(logits).all()
    finally:
        model.close()


def test_save_adapter_roundtrips(base_db):
    """The exported directory carries the factors `reminis lora` pairs up."""
    from reminis.lora import _split_lora_name
    from reminis.safetensors_io import load_tensors

    run = str(TMP / "run.db")
    meta = read_run_meta(run)
    conn = sqlite3.connect(run)
    try:
        out = save_adapter(conn, rank=int(meta["rank"]),
                           alpha=float(meta["alpha"]),
                           use_rslora=meta.get("use_rslora") == "true",
                           targets=meta["targets"].split(","), step=8,
                           out_dir=str(TMP / "adapter"), quiet=True)
    finally:
        conn.close()

    config = json.loads((Path(out) / "adapter_config.json").read_text())
    assert config["r"] == RANK and config["lora_alpha"] == ALPHA
    assert config["peft_type"] == "LORA"
    assert set(config["target_modules"]) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    conn = sqlite3.connect(run)
    state = {
        (tensor, role):
            np.frombuffer(blob, dtype=np.float32).reshape(json.loads(shape_json))
        for tensor, role, shape_json, blob in conn.execute(
            "SELECT tensor, role, shape, data FROM adapter_state WHERE step = 8")
    }
    conn.close()

    exported = load_tensors(str(Path(out) / "adapter_model.safetensors"))
    pairs = {}
    for name, (dtype, shape, blob) in exported.items():
        module, which = _split_lora_name(name)
        pairs.setdefault(module, {})[which.removeprefix("lora_")] = \
            np.frombuffer(blob, dtype=np.float32).reshape(shape)
    assert len(pairs) == len(state) // 2

    # Every exported factor is the checkpoint's, bit for bit, and at least
    # one B factor is nonzero: the run trained.
    nonzero = 0
    for (tensor, role), arr in state.items():
        module = tensor[:-len(".weight")]
        assert np.array_equal(pairs[module][role], arr), tensor
        if role == "B" and np.abs(arr).max() > 0:
            nonzero += 1
    assert nonzero > 0, "every B factor is zero; the run did not train"
