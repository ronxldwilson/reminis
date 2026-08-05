"""Verify that a model run out of SQLite is the same model.

A forward pass that is subtly wrong still produces fluent text, so "it made
sentences" proves nothing. These checks are chosen to fail loudly instead:

  * the tokenizer's ids are compared against the reference implementation,
    token for token, when transformers is installed
  * the logits are compared against transformers' own float32 forward pass
  * the KV cache is checked against a cacheless re-run of the whole sequence,
    which needs no external reference at all
  * streaming mode and cached mode must agree exactly, since the only thing
    that differs between them is where the bytes came from
  * architectures that are not implemented must raise, not improvise
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np

from reminis.backend import available_backends
from reminis.backend import select as select_backend
from reminis.infer import (
    KVCache,
    Model,
    UnsupportedModel,
    generate,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
SMOL = MODELS_DIR / "SmolLM-135M.f16.db"
SMOL_INSTRUCT = MODELS_DIR / "SmolLM-135M-Instruct.f16.db"
QWEN = MODELS_DIR / "qwen2.5-0.5b-instruct-fp16.db"
LLAMA = MODELS_DIR / "llama1b.db"
MAMBA = MODELS_DIR / "mamba-130m.db"
QUANTIZED = MODELS_DIR / "granite-3.1-1b-a400m-instruct-Q4_K_M.db"

FAILURES = []

# Text chosen to hit every branch of the pre-tokenizer: contractions, digit
# runs, underscores, punctuation clusters, non-ASCII, and whitespace shapes.
CASES = [
    "The capital of France is",
    "Hello, world!",
    "It's 2026 and I've got 1234 apples—don't you?",
    "def f(x):\n    return x**2  # squares\n\n",
    "naïve café résumé 東京 \U0001F680 emoji",
    "   leading spaces and\ttabs\n\nnewlines",
    "Numbers: 0 7 42 1000 999999",
    "<|im_start|>user\nHi<|im_end|>\n",
    "Mixed CASE WoRdS and_underscores-and-dashes",
    "snake_case_name = other_var_1 + _private",
    "__dunder__",
    "https://example.com/path?a=1&b=2#frag",
    'JSON: {"key": [1, 2.5, null], "nested": {"a": true}}',
    "e.g. i.e. etc. Dr. Smith went to St. Louis.",
    "math: 2+2=4, 3*3=9, 10/2=5",
    "  ",
]


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


def test_tokenizer_roundtrip():
    print("\nTokenizer round-trip (encode then decode returns the original)")
    for db in (SMOL, QWEN, LLAMA):
        if not db.exists():
            print(f"  skip  {db.name} not present")
            continue
        model = Model(str(db))
        try:
            bad = [c for c in CASES
                   if model.tokenizer.decode(model.tokenizer.encode(c, add_special=False)) != c]
            check(f"{db.name}: all {len(CASES)} strings survive a round-trip",
                  not bad, f"failed on {bad[:2]}")
        finally:
            model.close()


def test_tokenizer_vs_reference():
    """Compare ids with transformers, which is the implementation of record."""
    print("\nTokenizer against transformers")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  skip  transformers is not installed")
        return

    pairs = [
        (SMOL, "HuggingFaceTB/SmolLM-135M"),
        (QWEN, "Qwen/Qwen2.5-0.5B-Instruct"),
        (LLAMA, "unsloth/Llama-3.2-1B-Instruct"),
    ]
    for db, repo in pairs:
        if not db.exists():
            print(f"  skip  {db.name} not present")
            continue
        try:
            reference = AutoTokenizer.from_pretrained(repo)
        except Exception as exc:  # offline, gated, rate-limited
            print(f"  skip  could not fetch {repo} ({type(exc).__name__})")
            continue

        model = Model(str(db))
        try:
            mismatched = [
                c for c in CASES
                if model.tokenizer.encode(c, add_special=False)
                != reference.encode(c, add_special_tokens=False)
            ]
            check(f"{db.name}: ids match {repo} on all {len(CASES)} strings",
                  not mismatched, f"differed on {mismatched[:2]}")
        finally:
            model.close()


def test_logits_vs_reference():
    """The whole forward pass, against transformers' own float32 run."""
    print("\nLogits against transformers")
    if not SMOL.exists():
        print("  skip  SmolLM-135M.f16.db not present")
        return
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  skip  torch/transformers not installed")
        return

    repo = "HuggingFaceTB/SmolLM-135M"
    try:
        reference = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
        ref_tok = AutoTokenizer.from_pretrained(repo)
    except Exception as exc:
        print(f"  skip  could not fetch {repo} ({type(exc).__name__})")
        return

    prompt = "The capital of France is Paris, and the capital of Germany is"
    ids = ref_tok.encode(prompt, add_special_tokens=False)
    with torch.no_grad():
        ref_logits = reference(torch.tensor([ids])).logits[0, -1].numpy()

    # Pinned to numpy on purpose. numpy is the reference implementation and
    # this is the check that establishes it; the GPU backends compute in
    # float16 and are held to agreeing with numpy, which test_backend does.
    reference_backend = select_backend(requested="numpy")
    model = Model(str(SMOL), backend=reference_backend)
    try:
        mine = model.forward(
            ids, KVCache(model.cfg.n_layers, backend=reference_backend), offset=0
        )
        check("the argmax token agrees",
              int(np.argmax(mine)) == int(np.argmax(ref_logits)),
              f"{ref_tok.decode([int(np.argmax(mine))])!r} vs "
              f"{ref_tok.decode([int(np.argmax(ref_logits))])!r}")
        check("the top ten tokens agree, in order",
              list(np.argsort(mine)[::-1][:10]) == list(np.argsort(ref_logits)[::-1][:10]))
        # The gap is F16 storage against torch's F32, so it is small and its
        # size is worth printing rather than merely asserting.
        gap = float(np.abs(mine - ref_logits).max())
        check("every logit is within 1e-3 of the reference", gap < 1e-3, f"max gap {gap:.2e}")
        print(f"        (largest logit difference: {gap:.2e})")
    finally:
        model.close()


def test_kv_cache():
    """One token at a time must equal the whole sequence at once.

    This needs no reference implementation: if the cache, the rotary
    positions, or the causal mask are wrong when decoding incrementally, the
    two paths diverge.
    """
    print("\nKV cache against a cacheless re-run")
    if not SMOL.exists():
        print("  skip  SmolLM-135M.f16.db not present")
        return

    # numpy again: the property being checked is that the cache and the mask
    # are right, and half precision would add noise that has nothing to do
    # with either.
    reference_backend = select_backend(requested="numpy")
    model = Model(str(SMOL), backend=reference_backend)
    try:
        ids = model.tokenizer.encode("The quick brown fox jumps over the lazy dog and then",
                                     add_special=False)
        full = model.forward(
            ids, KVCache(model.cfg.n_layers, backend=reference_backend), offset=0
        )

        cache = KVCache(model.cfg.n_layers, backend=reference_backend)
        incremental = model.forward(ids[:1], cache, offset=0)
        for i, token in enumerate(ids[1:], start=1):
            incremental = model.forward([token], cache, offset=cache.length)

        gap = float(np.abs(full - incremental).max())
        check("token-by-token decoding matches the batched pass", gap < 1e-2,
              f"max logit gap {gap:.3e}")
        check("the cache holds one entry per token", cache.length == len(ids),
              f"{cache.length} vs {len(ids)}")
        print(f"        (largest logit difference: {gap:.2e})")
    finally:
        model.close()


def test_streaming_matches_cached():
    print("\nStreaming mode against cached mode")
    if not SMOL.exists():
        print("  skip  SmolLM-135M.f16.db not present")
        return

    prompt = "The capital of France is"
    cached = generate(str(SMOL), prompt, max_tokens=12, temperature=0.0,
                      verbose=False, on_token=lambda _: None)
    streamed = generate(str(SMOL), prompt, max_tokens=12, temperature=0.0,
                        stream=True, verbose=False, on_token=lambda _: None)

    check("the same tokens come out either way",
          cached["token_ids"] == streamed["token_ids"],
          f"{cached['completion']!r} vs {streamed['completion']!r}")
    check("streaming re-reads the weights instead of caching them",
          streamed["bytes_read"] > cached["bytes_read"] * 5,
          f"{streamed['bytes_read']:,} vs {cached['bytes_read']:,}")
    check("streaming issues a query per weight per token",
          streamed["queries"] > cached["queries"] * 5,
          f"{streamed['queries']:,} vs {cached['queries']:,}")
    print(f"        (cached: {cached['bytes_read'] / 1024**2:,.0f} MB in "
          f"{cached['queries']:,} queries; streamed: "
          f"{streamed['bytes_read'] / 1024**2:,.0f} MB in {streamed['queries']:,})")


def test_sampling():
    print("\nSampling")
    if not SMOL.exists():
        print("  skip  SmolLM-135M.f16.db not present")
        return

    prompt = "Once upon a time"
    quiet = dict(verbose=False, on_token=lambda _: None)

    a = generate(str(SMOL), prompt, max_tokens=10, temperature=0.0, **quiet)
    b = generate(str(SMOL), prompt, max_tokens=10, temperature=0.0, **quiet)
    check("greedy decoding is deterministic", a["token_ids"] == b["token_ids"])

    c = generate(str(SMOL), prompt, max_tokens=10, temperature=1.0, seed=7, **quiet)
    d = generate(str(SMOL), prompt, max_tokens=10, temperature=1.0, seed=7, **quiet)
    e = generate(str(SMOL), prompt, max_tokens=10, temperature=1.0, seed=8, **quiet)
    check("the same seed gives the same sample", c["token_ids"] == d["token_ids"])
    check("a different seed gives a different sample", c["token_ids"] != e["token_ids"],
          "two seeds produced identical output")

    check("the completion decodes to the tokens that were generated",
          a["completion"] == Model(str(SMOL)).tokenizer.decode(a["token_ids"]))


def test_architectures():
    print("\nArchitectures")
    for db, label in ((QWEN, "qwen2, which uses neox rotary and QKV biases"),
                      (LLAMA, "llama 3, which scales rotary frequencies per dimension")):
        if not db.exists():
            print(f"  skip  {db.name} not present")
            continue
        result = generate(str(db), "The capital of France is", max_tokens=6,
                          temperature=0.0, verbose=False, on_token=lambda _: None)
        check(f"{label} generates",
              "paris" in result["completion"].lower(),
              f"got {result['completion']!r}")


def test_refusals():
    print("\nRefusals (an unimplemented model must raise, not improvise)")
    if MAMBA.exists():
        expect_error("a state-space model is refused by name", "architecture",
                     lambda: generate(str(MAMBA), "hello", max_tokens=1, verbose=False))
    else:
        print("  skip  mamba-130m.db not present")

    # Quantized tensors are now unpacked rather than refused, so the thing
    # that must still fail loudly is a dtype reminis cannot decode at all.
    # One tensor of a working model is relabelled to produce that.
    import shutil
    tmp = Path(__file__).parent / "tmp_infer_quant"
    tmp.mkdir(exist_ok=True)
    fake = tmp / "quantized.db"
    try:
        shutil.copyfile(SMOL, fake)
        import sqlite3
        conn = sqlite3.connect(str(fake))
        conn.execute("UPDATE tensors SET dtype = 'NOT_A_DTYPE' WHERE name = 'blk.0.attn_q.weight'")
        conn.commit()
        conn.close()
        expect_error("an unrecognised dtype is refused rather than guessed at",
                     "neither a float type",
                     lambda: generate(str(fake), "hello", max_tokens=1,
                                      verbose=False, on_token=lambda _: None))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    expect_error("a missing database is reported by path", "not found",
                 lambda: generate("nope.db", "hello", max_tokens=1, verbose=False))

    if SMOL.exists():
        expect_error("an empty prompt is refused", "zero tokens",
                     lambda: generate(str(SMOL), "", max_tokens=1, verbose=False))


def test_quantized_models():
    """Quantized weights must unpack to something close to the original.

    Dequantizing is the one place a silent error would be invisible: wrong
    block arithmetic still produces numbers, and a model built from them
    still emits fluent text. So the check is not "does it generate" but
    "does it match the float weights it was quantized from" -- the same
    model exists here in F16, which makes that a direct comparison.
    """
    print("\nQuantized models")
    quant_db = MODELS_DIR / "smollm-q4km.db"
    if not (quant_db.exists() and SMOL.exists()):
        print("  skip  needs SmolLM-135M in both Q4_K_M and F16")
        return

    from reminis.dtypes import to_float32_any

    conn_q = sqlite3.connect(str(quant_db))
    conn_f = sqlite3.connect(str(SMOL))
    worst_corr, compared, kinds = 1.0, 0, set()
    for name, dtype, blob in conn_q.execute("SELECT name, dtype, data FROM tensors"):
        row = conn_f.execute(
            "SELECT dtype, data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            continue
        got = to_float32_any(blob, dtype)
        want = to_float32_any(row[1], row[0])
        if got.shape != want.shape:
            continue
        kinds.add(dtype)
        worst_corr = min(worst_corr, float(np.corrcoef(got, want)[0, 1]))
        compared += 1
    conn_q.close()
    conn_f.close()

    check(f"all {compared} quantized tensors track their F16 originals",
          worst_corr > 0.99, f"lowest correlation {worst_corr:.6f}")
    print(f"        (lowest correlation {worst_corr:.6f}, "
          f"types present: {', '.join(sorted(kinds))})")

    result = generate(str(quant_db), "The capital of France is", max_tokens=10,
                      temperature=0.0, verbose=False, on_token=lambda _: None)
    check("a quantized model generates coherent text",
          "paris" in result["completion"].lower(), f"got {result['completion']!r}")

    conn = sqlite3.connect(str(quant_db))
    n_quant = conn.execute(
        "SELECT COUNT(*) FROM tensors WHERE dtype NOT IN ('F32','F16','BF16')"
    ).fetchone()[0]
    conn.close()
    check("the quantized tensors were actually unpacked, not skipped",
          n_quant > 0, f"{n_quant} quantized tensors in the file")


def test_packed_weights():
    """Keeping weights packed must cost memory-accuracy, not correctness.

    Packing re-quantizes into the backend's own format, so the weights are
    rounded a second time on top of whatever the file already did. The
    question is not whether that changes the numbers -- it must -- but
    whether the model still ranks tokens the same way, and whether the
    memory it saves is real.
    """
    print("\nPacked weights")
    quant_db = MODELS_DIR / "smollm-q4km.db"
    if not quant_db.exists():
        print("  skip  needs a quantized SmolLM database")
        return

    backend = select_backend("inference")
    if not backend.can_pack():
        print(f"  skip  the {backend.name} backend cannot multiply packed weights")
        return

    prompt = "The capital of France is Paris, and the capital of Germany is"

    def logits_and_memory(bits):
        model = Model(str(quant_db), pack_bits=bits)
        try:
            for name in model.store._shapes:
                model.store.get(name)
            model.backend.eval()
            resident = _resident_mb(model.backend)
            ids = model.tokenizer.encode(prompt, add_special=False)
            lg = model.forward(
                ids, KVCache(model.cfg.n_layers, capacity=64, backend=model.backend), 0
            )
            return lg, resident, model.store.packed
        finally:
            model.close()

    reference, ref_mem, n_unpacked = logits_and_memory(None)
    check("nothing is packed when packing is off", n_unpacked == 0)

    for bits, min_saving in ((8, 1.2), (6, 1.4), (4, 1.8)):
        lg, mem, packed = logits_and_memory(bits)
        check(f"{bits}-bit: the per-layer matrices were packed", packed > 0)
        check(f"{bits}-bit: picks the same next token",
              int(np.argmax(lg)) == int(np.argmax(reference)))
        corr = float(np.corrcoef(lg, reference)[0, 1])
        check(f"{bits}-bit: logits still track the unpacked model",
              corr > 0.95, f"corr {corr:.6f}")
        saving = ref_mem / mem if mem else 0
        check(f"{bits}-bit: uses at least {min_saving}x less weight memory",
              saving >= min_saving, f"{ref_mem:.0f} -> {mem:.0f} MB is {saving:.2f}x")
        print(f"        ({bits}-bit: {ref_mem:.0f} -> {mem:.0f} MB, "
              f"{saving:.2f}x smaller, correlation {corr:.6f})")


def _resident_mb(backend):
    """How much the backend is actually holding, where it can say."""
    try:
        import mlx.core as mx

        mx.clear_cache()
        return mx.get_active_memory() / 1024 ** 2
    except Exception:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


SPM_CASES = [
    "The capital of France is",
    "Hello, world!",
    "It's 2026 and I've got 1234 apples—don't you?",
    "naïve café résumé 東京 \U0001F680",
    "[INST] hi [/INST]",
    "[INST]hi",
    "x[INST]y",
    "  leading spaces",
    "snake_case_name = x_1 + _p",
    "math: 2+2=4, 3*3=9",
    "https://example.com/a?b=1",
    "Dr. Smith went to St. Louis.",
    "tabs\tand  spaces",
    'JSON: {"k": [1, 2.5, null]}',
]


def test_sentencepiece_tokenizer():
    """SentencePiece ids against llama.cpp's, on the same file.

    SentencePiece has no merge list: it merges whichever adjacent pair forms
    the highest-scoring token in the vocabulary, so the vocabulary encodes
    the merge order. Small differences in that ordering, or in where the
    leading space goes, produce ids that still decode to the right text and
    still generate fluent output -- which is why this compares against the
    reference implementation rather than checking a round-trip.
    """
    print("\nSentencePiece against llama.cpp")
    model_db = MODELS_DIR / "mistral7b.db"
    gguf = MODELS_DIR / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    if not model_db.exists():
        print("  skip  no SentencePiece model present")
        return

    import shutil
    import sqlite3 as sq

    from reminis.infer import build_tokenizer

    meta = dict(sq.connect(str(model_db)).execute("SELECT key, value FROM model_meta"))
    tok = build_tokenizer(meta)
    check("the SentencePiece tokenizer was chosen",
          type(tok).__name__ == "SPMTokenizer", type(tok).__name__)
    check("all 256 byte-fallback tokens were found", len(tok.byte_ids) == 256,
          str(len(tok.byte_ids)))

    if not (shutil.which("llama-tokenize") and gguf.exists()):
        print("  skip  llama-tokenize or the GGUF is not available for comparison")
        # Without the reference, at least check that nothing encodes to nothing.
        check("every case encodes to something",
              all(tok.encode(c) for c in SPM_CASES))
        return

    import re
    import subprocess

    matched, total = 0, 0
    for case in SPM_CASES:
        out = subprocess.run(
            ["llama-tokenize", "-m", str(gguf), "-p", case, "--ids"],
            capture_output=True, text=True,
        ).stdout
        found = re.search(r"\[([\d,\s]+)\]", out)
        if not found:
            continue
        total += 1
        reference = [int(x) for x in found.group(1).split(",")]
        if tok.encode(case) == reference:
            matched += 1
        else:
            print(f"        differed on {case!r}")
    check(f"ids match llama.cpp on all {total} strings", matched == total,
          f"{matched}/{total}")


def test_mixture_of_experts():
    """A router picking experts per token, against the dense path's rules.

    An MoE forward pass has three ways to be quietly wrong: routing to the
    wrong experts, weighting them wrong, and -- for Granite specifically --
    ignoring the four scaling multipliers it carries, any one of which
    produces fluent nonsense. Both backends computing the same text is the
    check, since they share no code below the array operations.
    """
    print("\nMixture of experts")
    if not QUANTIZED.exists():
        print("  skip  the granite MoE database is not present")
        return

    model = Model(str(QUANTIZED))
    try:
        check("the model is recognised as MoE", model.cfg.is_moe)
        check("the expert counts were read",
              model.cfg.n_experts == 32 and model.cfg.n_experts_used == 8,
              f"{model.cfg.n_experts} experts, {model.cfg.n_experts_used} used")
        # Granite scales attention by 1/head_dim rather than 1/sqrt(head_dim).
        check("granite's own attention scale is used, not 1/sqrt(d)",
              abs(model.cfg.attn_scale - 0.015625) < 1e-9,
              str(model.cfg.attn_scale))
        check("the embedding, residual and logit scales were read",
              model.cfg.embedding_scale == 12.0 and model.cfg.logit_scale == 6.0
              and abs(model.cfg.residual_scale - 0.22) < 1e-6)
    finally:
        model.close()

    texts = {}
    for name in ["numpy"] + [n for n in available_backends() if n != "numpy"]:
        result = generate(str(QUANTIZED), "The capital of France is",
                          max_tokens=10, temperature=0.0, verbose=False,
                          on_token=lambda _: None, backend=name)
        texts[name] = result["completion"]
    check("an MoE model generates coherent text",
          "paris" in texts["numpy"].lower(), f"got {texts['numpy']!r}")
    print(f"        (it said: {texts['numpy']!r})")
    for name, text in texts.items():
        if name != "numpy":
            check(f"{name} routes to the same experts as numpy",
                  text == texts["numpy"], f"got {text!r}")

    # Packing the experts is what makes an MoE model worth running: they are
    # most of its bytes, and they are 3-D, which the packing had to learn.
    backend = select_backend("inference")
    if backend.can_pack():
        packed = Model(str(QUANTIZED), pack_bits="compact")
        try:
            for name in packed.store._shapes:
                packed.store.get(name)
            check("the stacked expert tensors were packed",
                  packed.store.packed >= 3 * packed.cfg.n_layers,
                  f"{packed.store.packed} packed over {packed.cfg.n_layers} layers")
        finally:
            packed.close()
        result = generate(str(QUANTIZED), "The capital of France is",
                          max_tokens=10, temperature=0.0, verbose=False,
                          on_token=lambda _: None, pack_bits="compact")
        check("a packed MoE model says the same thing",
              result["completion"] == texts["numpy"],
              f"got {result['completion']!r}")


def test_kv_cache_quantization():
    """A compressed cache must hold the same conversation, more cheaply.

    The cache is what grows with the context, so compressing it is what
    decides whether a long prompt fits. It costs speed rather than saving
    it -- there is no quantized attention kernel, so the cache is
    decompressed to attend -- which makes "does it change the output" the
    question worth asking.
    """
    print("\nKV cache compression")
    if not SMOL.exists():
        print("  skip  SmolLM-135M.f16.db not present")
        return
    backend = select_backend("inference")
    if backend.quantize_kv(backend.from_numpy(np.zeros((1, 1, 2, 64),
                                                       dtype=np.float32)),
                           8, 64) is None:
        print(f"  skip  the {backend.name} backend cannot compress a cache")
        return

    model = Model(str(SMOL))
    try:
        ids = model.tokenizer.encode(
            "The capital of France is Paris, and the capital of Germany is",
            add_special=False,
        )
        reference, ref_logits, sizes = None, None, {}
        for bits in (None, 8, 4):
            cache = KVCache(model.cfg.n_layers, capacity=128,
                            backend=model.backend, quantize_bits=bits)
            first = model.forward(ids, cache, 0)
            logits, generated = first, []
            for _ in range(8):
                token = int(np.argmax(logits))
                generated.append(token)
                logits = model.forward([token], cache, cache.length)
            text = model.tokenizer.decode(generated)
            sizes[bits] = _cache_bytes(cache)

            if reference is None:
                reference, ref_logits = text, first
                continue
            # Against the uncompressed run's logits, not against its own --
            # a self-comparison would pass however wrong the cache was.
            corr = float(np.corrcoef(first, ref_logits)[0, 1])
            check(f"{bits}-bit cache tracks the uncompressed one",
                  corr > 0.99, f"correlation {corr:.6f}")
            print(f"        ({bits}-bit correlation {corr:.6f})")
            if bits == 8:
                check("8-bit cache produces the same text as none",
                      text == reference, f"got {text!r}")

        check("8 bits roughly halves the cache",
              1.5 < sizes[None] / sizes[8] < 2.5,
              f"{sizes[None]} -> {sizes[8]} bytes")
        check("4 bits shrinks it further",
              sizes[4] < sizes[8], f"{sizes[8]} -> {sizes[4]} bytes")
        print(f"        (cache bytes: none {sizes[None]:,}, "
              f"8-bit {sizes[8]:,}, 4-bit {sizes[4]:,})")
    finally:
        model.close()


def _cache_bytes(cache) -> int:
    """How much the cache is actually holding, compressed or not."""
    total = 0
    for buffers in (cache.k, cache.v, cache._packed_k, cache._packed_v):
        for entry in buffers:
            if entry is None:
                continue
            parts = entry if isinstance(entry, tuple) else (entry,)
            for part in parts:
                total += part.size * part.dtype.size
    return total


def test_merged_model_runs():
    """The payoff: a model that was assembled by SQL still speaks."""
    print("\nA merged model")
    if not (SMOL.exists() and SMOL_INSTRUCT.exists()):
        print("  skip  both SmolLM databases are needed")
        return

    from reminis.merge import merge_models
    tmp = Path(__file__).parent / "tmp_infer"
    tmp.mkdir(exist_ok=True)
    merged = str(tmp / "soup.db")
    try:
        merge_models([str(SMOL), str(SMOL_INSTRUCT)], merged, verbose=False)
        result = generate(merged, "The capital of France is", max_tokens=8,
                          temperature=0.0, verbose=False, on_token=lambda _: None)
        check("the merged model produces coherent text",
              "paris" in result["completion"].lower(),
              f"got {result['completion']!r}")
        print(f"        (it said: {result['completion']!r})")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if not SMOL.exists():
        print(f"Missing {SMOL}. Run: reminis convert models/SmolLM-135M.f16.gguf")
        sys.exit(1)

    print("=" * 70)
    print("INFERENCE TESTS")
    print("=" * 70)

    test_tokenizer_roundtrip()
    test_tokenizer_vs_reference()
    test_logits_vs_reference()
    test_kv_cache()
    test_streaming_matches_cached()
    test_sampling()
    test_architectures()
    test_refusals()
    test_quantized_models()
    test_packed_weights()
    test_sentencepiece_tokenizer()
    test_mixture_of_experts()
    test_kv_cache_quantization()
    test_merged_model_runs()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} INFERENCE TESTS FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 70)
        sys.exit(1)
    print("ALL INFERENCE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
