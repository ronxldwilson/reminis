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

import sys
from pathlib import Path

import numpy as np

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

    model = Model(str(SMOL))
    try:
        mine = model.forward(ids, KVCache(model.cfg.n_layers), offset=0)
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

    model = Model(str(SMOL))
    try:
        ids = model.tokenizer.encode("The quick brown fox jumps over the lazy dog and then",
                                     add_special=False)
        full = model.forward(ids, KVCache(model.cfg.n_layers), offset=0)

        cache = KVCache(model.cfg.n_layers)
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

    if QUANTIZED.exists():
        expect_error("a state-space or MoE architecture is named in the refusal",
                     "granitemoe",
                     lambda: generate(str(QUANTIZED), "hello", max_tokens=1,
                                      verbose=False, on_token=lambda _: None))

    # A quantized *llama* is the case that matters, since it gets past the
    # architecture check and would otherwise be decoded as though its Q4_K
    # blocks were floats. Rather than keep a second quantized model around,
    # one tensor of a working model is relabelled.
    import shutil
    tmp = Path(__file__).parent / "tmp_infer_quant"
    tmp.mkdir(exist_ok=True)
    fake = tmp / "quantized.db"
    try:
        shutil.copyfile(SMOL, fake)
        import sqlite3
        conn = sqlite3.connect(str(fake))
        conn.execute("UPDATE tensors SET dtype = 'Q4_K' WHERE name = 'blk.0.attn_q.weight'")
        conn.commit()
        conn.close()
        expect_error("a quantized tensor is refused before it produces noise",
                     "quantized",
                     lambda: generate(str(fake), "hello", max_tokens=1,
                                      verbose=False, on_token=lambda _: None))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    expect_error("a missing database is reported by path", "not found",
                 lambda: generate("nope.db", "hello", max_tokens=1, verbose=False))

    if SMOL.exists():
        expect_error("an empty prompt is refused", "zero tokens",
                     lambda: generate(str(SMOL), "", max_tokens=1, verbose=False))


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
