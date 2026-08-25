"""A safetensors model must convert into a database that runs.

Converting stores the weights whatever happens. Running them needs three
more things, and each was missing until the architecture was taught to
recognise its own config.json:

  - `general.architecture` as the name reminis uses, not the HuggingFace
    class name, or nothing resolves at all
  - the GGUF-style hyperparameters `ModelConfig` reads, since two of its
    fallbacks (rope base, RMS epsilon) are wrong for Qwen2 in a way that
    produces slightly-wrong text rather than an error
  - tensors under the names the model asks for

So this builds a small but complete Qwen2, converts it, and checks all
three -- ending with a forward pass, which is the only one of the four
that fails if any of the others is subtly wrong.
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reminis import arch as arch_registry
from reminis.kvcache import KVCache
from reminis.model import Model
from reminis.safetensors_io import safetensors_to_sqlite

try:
    import torch
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover - fixtures need the real libraries
    import pytest

    pytest.skip("install `torch` and `safetensors`", allow_module_level=True)

VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, FFN = 64, 32, 2, 4, 2, 64
HEAD_DIM = HIDDEN // HEADS
KV_WIDTH = KV_HEADS * HEAD_DIM


def build_qwen2(directory: Path) -> None:
    """A Qwen2 small enough to run in a test and complete enough to run."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    def t(*shape):
        # Small values keep the un-normalised logits finite in float16.
        return (torch.randn(*shape) * 0.02).to(torch.float32)

    tensors = {"model.embed_tokens.weight": t(VOCAB, HIDDEN)}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        tensors.update({
            # Qwen2 carries biases on the query, key and value projections.
            p + "self_attn.q_proj.weight": t(HIDDEN, HIDDEN),
            p + "self_attn.q_proj.bias": t(HIDDEN),
            p + "self_attn.k_proj.weight": t(KV_WIDTH, HIDDEN),
            p + "self_attn.k_proj.bias": t(KV_WIDTH),
            p + "self_attn.v_proj.weight": t(KV_WIDTH, HIDDEN),
            p + "self_attn.v_proj.bias": t(KV_WIDTH),
            p + "self_attn.o_proj.weight": t(HIDDEN, HIDDEN),
            p + "mlp.gate_proj.weight": t(FFN, HIDDEN),
            p + "mlp.up_proj.weight": t(FFN, HIDDEN),
            p + "mlp.down_proj.weight": t(HIDDEN, FFN),
            p + "input_layernorm.weight": torch.ones(HIDDEN),
            p + "post_attention_layernorm.weight": torch.ones(HIDDEN),
        })
    tensors["model.norm.weight"] = torch.ones(HIDDEN)

    save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})

    (directory / "config.json").write_text(json.dumps({
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "intermediate_size": FFN,
        "vocab_size": VOCAB,
        # Both differ from the reader's fallback, which is why they have to
        # survive the trip through config.json.
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 128,
        "tie_word_embeddings": True,
    }, indent=2))

    # A minimal byte-level BPE, which is the one family the reader ingests.
    # It needs a non-empty merge list to be taken at all.
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


def test_hf_name_resolves():
    """Qwen2's config.json must name an architecture reminis knows."""
    name, spec = arch_registry.from_hf_name("Qwen2ForCausalLM")
    assert name == "qwen2", f"expected qwen2, got {name!r}"
    assert spec is not None
    # An unset pre-tokenizer silently falls back to GPT-2's splitter, which
    # tokenises this model wrongly without ever erroring.
    assert spec.pretokenizer == "qwen2"


def test_translate_name():
    """The llama-family block, in both spellings."""
    spec = arch_registry.get("qwen2")
    cases = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
        "model.layers.0.self_attn.q_proj.weight": "blk.0.attn_q.weight",
        "model.layers.7.self_attn.q_proj.bias": "blk.7.attn_q.bias",
        "model.layers.3.self_attn.o_proj.weight": "blk.3.attn_output.weight",
        "model.layers.1.mlp.gate_proj.weight": "blk.1.ffn_gate.weight",
        "model.layers.1.mlp.down_proj.weight": "blk.1.ffn_down.weight",
        "model.layers.2.input_layernorm.weight": "blk.2.attn_norm.weight",
        "model.layers.2.post_attention_layernorm.weight": "blk.2.ffn_norm.weight",
    }
    for hf, expected in cases.items():
        assert spec.translate_name(hf) == expected, hf

    # Nothing recognised is left alone rather than guessed at, so a missing
    # tensor is reported under the name the model asked for.
    assert spec.translate_name("model.layers.0.something_new.weight") is None
    assert spec.translate_name("not.a.model.tensor") is None


def test_converted_model_runs(tmp):
    """The whole chain: convert, then generate logits from the rows."""
    model_dir = Path(tmp) / "qwen2_tiny"
    db_path = Path(tmp) / "qwen2_tiny.db"
    build_qwen2(model_dir)
    safetensors_to_sqlite(str(model_dir), str(db_path), verbose=False)

    conn = sqlite3.connect(db_path)
    meta = dict(conn.execute("SELECT key, value FROM model_meta").fetchall())
    names = {r[0] for r in conn.execute("SELECT name FROM tensors").fetchall()}
    conn.close()

    assert meta["general.architecture"] == "qwen2"

    # The hyperparameters the reader needs. The last two would otherwise
    # fall back to values that are wrong for this model.
    assert meta["qwen2.block_count"] == str(LAYERS)
    assert meta["qwen2.embedding_length"] == str(HIDDEN)
    assert meta["qwen2.attention.head_count"] == str(HEADS)
    assert meta["qwen2.attention.head_count_kv"] == str(KV_HEADS)
    assert float(meta["qwen2.rope.freq_base"]) == 1000000.0
    assert float(meta["qwen2.attention.layer_norm_rms_epsilon"]) == 1e-06

    assert "token_embd.weight" in names
    assert "output_norm.weight" in names
    assert "blk.0.attn_q.weight" in names
    assert "blk.0.attn_q.bias" in names
    assert "blk.1.ffn_gate.weight" in names
    assert not any(n.startswith("model.layers.") for n in names), \
        "untranslated HuggingFace names left in the database"

    model = Model(str(db_path))
    try:
        assert model.cfg.n_layers == LAYERS
        assert model.cfg.n_heads == HEADS
        assert model.cfg.rope_base == 1000000.0

        tokens = [1, 2, 3, 4]
        logits = np.asarray(
            model.forward(tokens, KVCache(model.cfg.n_layers), 0)
        ).ravel()
        assert logits.shape[0] == VOCAB, logits.shape
        assert np.isfinite(logits).all(), "forward pass produced non-finite logits"
    finally:
        model.close()
