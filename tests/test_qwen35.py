"""Verify the hybrid recurrent-attention architecture, without a 10 GB file.

qwen35 is three-quarters Gated DeltaNet and one-quarter attention, and the
recurrent part is where it can go wrong in ways that still produce fluent
text. A published model is far too large to keep beside the suite, so this
builds a four-layer one with the same *shape* -- the same fused projections,
the same head geometry, the same one-in-four attention -- and checks the
properties that do not need a reference implementation to state:

  * a recurrent layer carries state between calls, so reading a prompt in
    one call, a token at a time, and in two chunks must all agree. Nothing
    else in the output would reveal a dropped conv window or a state that
    is rebuilt from nothing each time.
  * the two backends must agree, since only the arithmetic library differs
  * the multi-token-prediction head is not a layer of the stack and must be
    left out of it
  * a decay that would make the state grow must be refused rather than run

The weights are random, so what comes out is gibberish either way -- which
is the point. Every check here is about internal consistency, and each one
fails loudly for a different reason.
"""


import numpy as np

from reminis.backend import available_backends
from reminis.backend import select as select_backend
from reminis.converter import gguf_to_sqlite
from reminis.infer import KVCache, Model, UnsupportedModel

FAILURES = []

# Small enough to build in a moment, but every relation the real model has
# is preserved: the fused projection is 2 * key + value wide, the value
# heads number inner / state, and the key heads number `group_count`.
D_MODEL = 64
N_LAYERS = 4          # layer 3 attends, 0-2 recur
N_HEADS = 4
HEAD_DIM = 16
N_KV_HEADS = 2
STATE = 8
GROUPS = 2
INNER = 32
DT_RANK = INNER // STATE
FFN = 128
ROPE_DIM = 8
VOCAB = 32
CONV_K = 4
K_DIM = GROUPS * STATE
QKV_DIM = 2 * K_DIM + INNER

TOKENS = [5, 9, 3, 12, 7, 2]


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"  --  {detail}" if detail else ""))
        FAILURES.append(label)


def expect_error(label, fragment, fn):
    try:
        fn()
    except (UnsupportedModel, ValueError, FileNotFoundError) as exc:
        check(label, fragment.lower() in str(exc).lower(), f"message was: {exc}")
        return
    check(label, False, "no error raised")


def build_gguf(path, ssm_a_sign=-1.0):
    """A four-layer qwen35 whose shapes match a published one's.

    `ssm_a_sign` exists so the divergence check can be given the file it is
    meant to reject: a trained ssm_a is negative, and a positive one is the
    sign convention error that would otherwise pass silently.
    """
    from gguf import GGUFWriter

    rng = np.random.default_rng(0)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    def ones(*shape):
        return np.ones(shape, dtype=np.float32)

    w = GGUFWriter(str(path), "qwen35")
    # One more block than the stack runs: the last is the prediction head.
    w.add_uint32("qwen35.block_count", N_LAYERS + 1)
    w.add_uint32("qwen35.nextn_predict_layers", 1)
    w.add_uint32("qwen35.context_length", 256)
    w.add_uint32("qwen35.embedding_length", D_MODEL)
    w.add_uint32("qwen35.feed_forward_length", FFN)
    w.add_uint32("qwen35.attention.head_count", N_HEADS)
    w.add_uint32("qwen35.attention.head_count_kv", N_KV_HEADS)
    w.add_uint32("qwen35.attention.key_length", HEAD_DIM)
    w.add_uint32("qwen35.attention.value_length", HEAD_DIM)
    w.add_float32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_float32("qwen35.rope.freq_base", 10000.0)
    w.add_uint32("qwen35.rope.dimension_count", ROPE_DIM)
    w.add_uint32("qwen35.ssm.conv_kernel", CONV_K)
    w.add_uint32("qwen35.ssm.state_size", STATE)
    w.add_uint32("qwen35.ssm.group_count", GROUPS)
    w.add_uint32("qwen35.ssm.time_step_rank", DT_RANK)
    w.add_uint32("qwen35.ssm.inner_size", INNER)
    w.add_uint32("qwen35.full_attention_interval", 4)

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list([chr(65 + i) for i in range(VOCAB)])
    w.add_token_types([1] * VOCAB)
    w.add_token_merges(["A B"])
    w.add_bos_token_id(0)
    w.add_eos_token_id(1)

    w.add_tensor("token_embd.weight", r(VOCAB, D_MODEL))
    w.add_tensor("output_norm.weight", ones(D_MODEL))
    w.add_tensor("output.weight", r(VOCAB, D_MODEL))

    for i in range(N_LAYERS):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", ones(D_MODEL))
        w.add_tensor(p + "post_attention_norm.weight", ones(D_MODEL))
        w.add_tensor(p + "ffn_gate.weight", r(FFN, D_MODEL))
        w.add_tensor(p + "ffn_up.weight", r(FFN, D_MODEL))
        w.add_tensor(p + "ffn_down.weight", r(D_MODEL, FFN))

        if (i + 1) % 4 != 0:
            w.add_tensor(p + "attn_qkv.weight", r(QKV_DIM, D_MODEL))
            w.add_tensor(p + "attn_gate.weight", r(INNER, D_MODEL))
            w.add_tensor(p + "ssm_conv1d.weight", r(CONV_K, QKV_DIM))
            w.add_tensor(p + "ssm_a", (ssm_a_sign * (
                np.abs(rng.standard_normal(DT_RANK)) + 0.5)).astype(np.float32))
            w.add_tensor(p + "ssm_alpha.weight", r(DT_RANK, D_MODEL))
            w.add_tensor(p + "ssm_beta.weight", r(DT_RANK, D_MODEL))
            w.add_tensor(p + "ssm_dt.bias", r(DT_RANK))
            w.add_tensor(p + "ssm_norm.weight", ones(STATE))
            w.add_tensor(p + "ssm_out.weight", r(D_MODEL, INNER))
        else:
            w.add_tensor(p + "attn_q.weight", r(2 * N_HEADS * HEAD_DIM, D_MODEL))
            w.add_tensor(p + "attn_k.weight", r(N_KV_HEADS * HEAD_DIM, D_MODEL))
            w.add_tensor(p + "attn_v.weight", r(N_KV_HEADS * HEAD_DIM, D_MODEL))
            w.add_tensor(p + "attn_q_norm.weight", ones(HEAD_DIM))
            w.add_tensor(p + "attn_k_norm.weight", ones(HEAD_DIM))
            w.add_tensor(p + "attn_output.weight", r(D_MODEL, N_HEADS * HEAD_DIM))

    # The prediction head, which the stack must skip. Its tensors are the
    # only ones it has, so running it as an ordinary layer would fail.
    n = f"blk.{N_LAYERS}."
    w.add_tensor(n + "nextn.eh_proj.weight", r(D_MODEL, 2 * D_MODEL))
    w.add_tensor(n + "nextn.enorm.weight", ones(D_MODEL))
    w.add_tensor(n + "nextn.hnorm.weight", ones(D_MODEL))
    w.add_tensor(n + "nextn.shared_head_norm.weight", ones(D_MODEL))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def build_db(tmp, name="qwen35.db", ssm_a_sign=-1.0):
    gguf = tmp / "qwen35.gguf"
    db = tmp / name
    for path in (gguf, db):
        if path.exists():
            path.unlink()
    build_gguf(gguf, ssm_a_sign)
    gguf_to_sqlite(str(gguf), str(db), verbose=False)
    return db


def _logits(db, backend, plan):
    """Final-position logits, reading the prompt in the chunks `plan` names."""
    model = Model(str(db), backend=select_backend("inference", backend))
    cache = KVCache(model.cfg.n_layers, capacity=len(TOKENS) + 4,
                    backend=model.backend)
    out, offset = None, 0
    for chunk in plan:
        out = model.forward(TOKENS[offset:offset + chunk], cache, offset=offset,
                            all_positions=True)
        offset += chunk
    model.close()
    return np.asarray(out)[-1]


def test_recurrent_state_carries(tmp):
    print("\nRecurrent state across calls (no reference implementation needed)")
    db = build_db(tmp)
    n = len(TOKENS)
    for backend in available_backends():
        whole = _logits(db, backend, [n])
        # float16 backends carry more rounding than float32 ones, and the
        # tolerance is about the arithmetic rather than about this test.
        tol = 5e-3 if backend != "numpy" else 1e-5
        for label, plan in (("one token at a time", [1] * n),
                            ("two chunks", [4, n - 4])):
            other = _logits(db, backend, plan)
            scale = max(float(np.max(np.abs(whole))), 1e-9)
            rel = float(np.max(np.abs(whole - other))) / scale
            check(f"{backend}: whole prompt == {label}", rel < tol,
                  f"relative difference {rel:.2e}")


def test_backends_agree(tmp):
    print("\nBackends agree (only the array library differs)")
    backends = available_backends()
    if len(backends) < 2:
        print(f"  skip  only {backends[0]} is available here")
        return
    db = build_db(tmp)
    reference = _logits(db, "numpy", [len(TOKENS)])
    for backend in backends:
        if backend == "numpy":
            continue
        other = _logits(db, backend, [len(TOKENS)])
        scale = max(float(np.max(np.abs(reference))), 1e-9)
        rel = float(np.max(np.abs(reference - other))) / scale
        check(f"numpy == {backend}", rel < 5e-3, f"relative difference {rel:.2e}")
        check(f"numpy and {backend} rank the same token first",
              int(np.argmax(reference)) == int(np.argmax(other)))


def test_prediction_head_skipped(tmp):
    print("\nThe multi-token-prediction head is not a layer of the stack")
    db = build_db(tmp)
    model = Model(str(db), backend=select_backend("inference", "numpy"))
    n_layers = model.cfg.n_layers
    is_ssm = list(model.cfg.is_ssm_layer)
    model.close()
    check("block_count counts the head, the stack does not",
          n_layers == N_LAYERS, f"ran {n_layers} layers, expected {N_LAYERS}")
    check("one layer in four attends, the rest recur",
          is_ssm == [True, True, True, False], f"got {is_ssm}")


def test_delta_rule_matches_reference():
    """The recurrence against a transcription of the reference loop.

    Every other check in this file tests the implementation against itself
    -- that state carries, that backends agree, that shapes line up. All of
    those passed while the recurrence was the wrong one: a gated linear
    attention rather than a delta rule, which on the real weights produced
    the same token forever and no error at all.

    So this one holds the arithmetic against the reference's own loop,
    written out below in the order that implementation performs it. The
    check has teeth: the rule that was here before differs from this by
    about a third of the signal, not by a rounding.
    """
    print("\nRecurrence vs the reference implementation")
    H, D, T = 4, 8, 6
    rng = np.random.default_rng(0)

    def unit(x):
        return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)

    q = unit(rng.standard_normal((T, H, D)).astype(np.float32))
    k = unit(rng.standard_normal((T, H, D)).astype(np.float32))
    v = rng.standard_normal((T, H, D)).astype(np.float32)
    g = -np.abs(rng.standard_normal((T, H)).astype(np.float32)) * 0.3
    beta = rng.random((T, H)).astype(np.float32)

    # The recurrence as llama.cpp's GATED_DELTA_NET kernel performs it,
    # which is the reference that matters here because the weights arrive
    # as GGUF. It agrees with modeling_qwen3_next's loop except for the
    # last line: the kernel scales the readout by 1/sqrt(head_dim) and
    # leaves the state unscaled.
    scale = 1.0 / np.sqrt(D)
    state = np.zeros((H, D, D), dtype=np.float32)      # (key, value)
    reference = []
    for i in range(T):
        state = state * np.exp(g[i])[:, None, None]
        kv_mem = (state * k[i][:, :, None]).sum(axis=-2)
        delta = (v[i] - kv_mem) * beta[i][:, None]
        state = state + k[i][:, :, None] * delta[:, None, :]
        reference.append((state * q[i][:, :, None]).sum(axis=-2) * scale)
    reference = np.stack(reference).reshape(T, H * D)

    from reminis import arch as arch_registry

    holder = type("_M", (), {})()
    holder.backend = select_backend("inference", "numpy")
    got, _ = arch_registry.get("qwen35")._deltanet_scan(
        holder, q, k, v, g, beta, None)
    got = np.asarray(got)

    scale = max(float(np.max(np.abs(reference))), 1e-9)
    rel = float(np.max(np.abs(reference - got))) / scale
    check("the recurrence reproduces the reference", rel < 1e-5,
          f"relative difference {rel:.2e}")

    # The rule this replaced, to show the comparison can fail.
    naive = np.zeros((H, D, D), dtype=np.float32)
    wrong = []
    for i in range(T):
        naive = (np.exp(g[i])[:, None, None] * naive
                 + beta[i][:, None, None] * (v[i][:, :, None] * k[i][:, None, :]))
        wrong.append((naive @ q[i][:, :, None])[..., 0])
    wrong = np.stack(wrong).reshape(T, H * D)
    gap = float(np.max(np.abs(reference - wrong))) / scale
    check("a rule without the delta correction would be caught", gap > 1e-2,
          f"the two rules differ by only {gap:.2e}")


def test_pretokenizer():
    """The splitter is this model's own, not the one it falls back to.

    An unrecognised pre-tokenizer name silently becomes GPT-2's, and the
    two disagree on every number: qwen35 takes digits one at a time where
    GPT-2 takes a whole run. Nothing downstream notices -- the ids are
    valid, they are simply not the ids the model was trained on -- so this
    checks the split directly rather than trusting the lookup.
    """
    print("\nPre-tokenizer (a wrong splitter is silent, so check the splits)")
    import regex

    from reminis.infer import _PRETOKENIZERS, _QWEN35_PATTERN

    pattern = regex.compile(_QWEN35_PATTERN)
    check("qwen35 is not left to the default splitter",
          "qwen35" not in _PRETOKENIZERS,
          "it is in the plain-`re` table, where its \\p{...} classes cannot work")
    check("digits split one at a time",
          pattern.findall("2026") == ["2", "0", "2", "6"],
          f"got {pattern.findall('2026')}")
    check("a combining mark stays with its letter",
          pattern.findall("ábc") == ["ábc"],
          f"got {pattern.findall('a' + chr(0x301) + 'bc')}")
    check("contractions attach to the quote",
          pattern.findall("don't") == ["don", "'t"],
          f"got {pattern.findall(chr(39).join(['don', 't']))}")


def test_divergent_decay_refused(tmp):
    print("\nA decay that grows the state is refused, not run")
    db = build_db(tmp, name="qwen35_bad.db", ssm_a_sign=+1.0)
    expect_error(
        "positive ssm_a is refused by name", "ssm_a is not negative",
        lambda: Model(str(db), backend=select_backend("inference", "numpy")),
    )
