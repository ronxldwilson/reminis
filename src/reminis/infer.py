"""Run a model straight out of its database.

Everything a forward pass needs is already in the file: the weights are rows
in ``tensors``, the hyperparameters and the tokenizer's vocabulary and merges
are rows in ``model_meta``. So this module loads no config, downloads nothing,
and imports neither torch nor llama.cpp -- it is numpy over the result of a
``SELECT``.

The point is not speed. It is that a reminis database is a *complete*,
self-contained model rather than an archive of one: if it can generate text,
then the conversion kept everything that mattered, and a merged or
rolled-back or delta-applied database can be checked by asking it to speak
rather than by comparing hashes.

Speed, measured rather than assumed, turns out to be less embarrassing than
expected -- 0.86-0.89x llama.cpp's CPU token generation across three model
sizes, and faster than it at prompt processing, since both end up in the
platform BLAS for the large matrix multiplies. Against llama.cpp on the GPU
it is 2.6-2.8x slower.

``--stream`` takes that further. In streaming mode no weight is ever cached:
every matrix multiplication in every layer re-reads its operand from SQLite
and throws it away, so peak memory is one layer rather than one model. It is
slow, and it demonstrates the thing the whole project is about -- that the
model is data in a database, paged in on demand, not a file that must fit in
RAM.

Scope, deliberately narrow and loudly enforced:

  * llama-family and qwen2 architectures from GGUF (rotary, RMSNorm, SwiGLU,
    grouped-query attention), which covers llama, qwen2, smollm, mistral
  * float weights -- F32, F16, BF16
  * quantized weights, unpacked at load through the ``gguf`` package: every
    K-quant and i-quant llama.cpp writes. Note what this is not -- the blocks
    become float16 in memory, so a quantized model becomes *runnable*, not
    *small*. Quantization saves the file and the download here, not the RAM.
  * a byte-level BPE tokenizer stored in the database

Anything else raises rather than approximates, because a forward pass that
guesses produces fluent-looking nonsense, which is worse than an error.

What is left in this file is the loop: sample a token, append it, do it
again. The pieces it drives each got their own module once this one reached
2,600 lines, since "everything inference needs" had stopped being a subject:

  ``weights.py``    a row becomes something that multiplies
  ``config.py``     rows become the handful of numbers a forward pass needs
  ``tokenizer.py``  text becomes ids, using only what the database holds
  ``model.py``      the forward pass
  ``kvcache.py``    keys and values, kept across steps
  ``meta.py``       reading list-shaped values back out of ``model_meta``
  ``errors.py``     the one exception all of them raise

They are re-exported below, so ``from reminis.infer import Model`` keeps
working and no caller had to change.
"""

import sys
import time
from pathlib import Path

import numpy as np

from reminis.backend import select as select_backend
from reminis.errors import UnsupportedModel
from reminis.kvcache import KVCache
from reminis.model import Model

# Re-exported, not used here. `from reminis.infer import X` was the only way
# to reach any of this before the split, so the names stay reachable at the
# old address -- including the underscored ones, which the suites import by
# name to test a pre-tokenizer pattern or a rotary period directly.
from reminis.backend import best_group as _best_group  # noqa: F401
from reminis.config import SUPPORTED_ARCHS, Config, _yarn_periods  # noqa: F401
from reminis.meta import _int_or_none, _parse_array  # noqa: F401
from reminis.tokenizer import (  # noqa: F401
    _GPT4O_PATTERN,
    _PRETOKENIZERS,
    _QWEN35_PATTERN,
    BPETokenizer,
    PieceBPETokenizer,
    SPMTokenizer,
    build_tokenizer,
)
from reminis.weights import WeightStore  # noqa: F401

__all__ = [
    "BPETokenizer",
    "Config",
    "KVCache",
    "Model",
    "PieceBPETokenizer",
    "SPMTokenizer",
    "SUPPORTED_ARCHS",
    "UnsupportedModel",
    "WeightStore",
    "build_tokenizer",
    "generate",
    "run_cli",
]

# ---------------------------------------------------------------- sampling


def _probs(logits: np.ndarray, temperature: float, top_p: float) -> np.ndarray:
    """The distribution the next token is drawn from, over the whole vocabulary.

    Kept as a full-width vector with the tokens outside the nucleus set to
    zero, rather than as a shortlist. Sampling only needs the shortlist,
    but speculative decoding has to compare the target's distribution with
    the draft's token by token, and two shortlists of different tokens do
    not line up. One vector per vocabulary is a megabyte and the comparison
    is then an array subtraction.
    """
    scaled = logits.astype(np.float32) / np.float32(temperature)
    scaled = scaled - np.max(scaled)
    np.exp(scaled, out=scaled)
    probs = scaled / np.sum(scaled)

    if 0 < top_p < 1:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        # Keep the smallest set of tokens whose mass reaches top_p, always
        # including the first one so the set is never empty.
        keep = int(np.searchsorted(cumulative, top_p) + 1)
        probs[order[keep:]] = 0.0
        probs /= probs.sum()

    return probs


def _sample(logits: np.ndarray, temperature: float, top_p: float, rng) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    probs = _probs(logits, temperature, top_p)
    return int(rng.choice(len(probs), p=probs))


def generate(
    db_path: str,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int | None = None,
    stream: bool = False,
    chat: bool = False,
    think: bool = False,
    stop_at_eos: bool = True,
    verbose: bool = True,
    on_token=None,
    backend: str | None = None,
    pack_bits=None,
    kv_bits: int | None = None,
    expert_cache: int = 0,
    expert_bits: int | None = None,
    preload: bool = False,
    draft: str | None = None,
    draft_tokens: int = 4,
) -> dict:
    """Generate text from a model stored in a reminis database.

    Args:
        db_path: The model database.
        prompt: The prompt text.
        max_tokens: How many tokens to generate at most.
        temperature: 0 is greedy; higher is more random.
        top_p: Nucleus sampling cutoff. 1 disables it.
        seed: Seed for sampling, so a run can be repeated exactly.
        kv_bits: Compress the key/value cache to this many bits. The cache
            grows with the context where the weights do not, so this is
            what decides whether a long prompt fits. It costs speed rather
            than saving it -- there is no quantized attention kernel, so
            the cache is decompressed to attend.
        pack_bits: Keep the big per-layer matrices packed at this many bits
            instead of unpacking them to float16, on a backend that can
            multiply them packed. Trades accuracy for memory: 6 keeps the
            top-5 ranking intact for 1.7x less, 4 goes to 2.1x less and
            visibly reorders it.
        stream: Re-read every weight from SQLite instead of caching it, so
            peak memory is one layer rather than the whole model.
        chat: Wrap the prompt in the model's chat template, when it has a
            ChatML one.
        think: On a reasoning model, leave its thinking channel open rather
            than closing it immediately. The answer then arrives after the
            working, which needs a token budget to match.
        draft: Propose tokens with something cheap and check them against
            this model in batches, which is faster because checking five
            tokens reads the weights once where producing five reads them
            five times. Either "ngram", to draft from the context with no
            second model, or the path to a smaller model that shares this
            one's tokenizer. The output is the same either way -- what
            changes is how many passes it takes.
        draft_tokens: How many tokens to propose per round. Too few wastes
            the batch; too many spends draft time on proposals that a
            rejection earlier in the round throws away.
        stop_at_eos: Stop when the model emits its end-of-text token.
        verbose: Print the header, the prompt, and the closing timings. The
            generated text itself is printed either way, so that piping the
            output somewhere gives the completion and nothing else.
        on_token: Optional callback, called with each decoded token string.
            Supplying one turns off printing, since the caller is handling it.

    Returns:
        A dict with the prompt, the completion, and timing.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    chosen = select_backend("inference", backend)
    model = Model(db_path, stream=stream, backend=chosen, pack_bits=pack_bits,
                  expert_cache=expert_cache, expert_bits=expert_bits)
    if preload:
        if verbose:
            print("Loading the expert index into memory", end="", flush=True)
        loaded = model.preload_experts()
        if verbose:
            print(f"\r{loaded['experts']:,} experts resident "
                  f"({loaded['bytes'] / 1e9:.2f} GB) in "
                  f"{loaded['seconds']:.0f}s")
    tok = model.tokenizer
    rng = np.random.default_rng(seed)
    speculator = None

    try:
        text = _apply_chat_template(prompt, tok, think) if chat else prompt
        tokens = tok.encode(text)
        if not tokens:
            raise ValueError("The prompt encoded to zero tokens")
        if len(tokens) >= model.cfg.context_length:
            raise ValueError(
                f"The prompt is {len(tokens)} tokens and this model's context "
                f"is {model.cfg.context_length}"
            )

        if verbose:
            mode = "streaming from SQLite" if stream else "weights cached in RAM"
            print(f"{model.meta.get('general.name', Path(db_path).name)} "
                  f"| {model.cfg.arch} | {model.cfg.n_layers} layers | "
                  f"{chosen.describe()} | {mode}")
            print(f"{len(tokens)} prompt tokens\n")
            print(text, end="", flush=True)

        if pack_bits is not None and not chosen.can_pack():
            print(f"Note: the {chosen.name} backend cannot multiply packed "
                  f"weights, so --pack was ignored.")

        # A packed index holds one width. Honouring `--pack` against it would
        # mean rebuilding every weight, which is the work the index exists to
        # have already done -- so the index wins and says so. Silence here
        # would mean asking for four bits and running three.
        index_bits = model.store.index_bits
        if index_bits is not None and pack_bits is not None:
            asked = pack_bits if isinstance(pack_bits, int) else None
            if asked != index_bits:
                wanted = (f"--pack {asked}" if asked is not None
                          else "bit-exact packing")
                print(f"Note: this database has a packed index built at "
                      f"{index_bits} bits, and it is being used. {wanted} "
                      f"would mean rebuilding every weight, which is what "
                      f"the index is there to avoid.\n"
                      f"      To run at another width: reminis prepare "
                      f"<db> --weights --bits N   (or --weights --drop)")

        # Speculation overruns: a round reads `draft_tokens` past the end
        # before finding out how many of them were wanted, and on a model
        # whose state cannot be rolled back it re-reads a queue of up to
        # 2(k+1) accepted tokens alongside them. The cache is given room
        # for the worst of that rather than doubling in the middle of a
        # round -- it would still be correct, only slower and larger.
        headroom = 3 * draft_tokens + 5 if draft else 0
        capacity = len(tokens) + max_tokens + headroom
        cache = KVCache(model.cfg.n_layers, capacity=capacity,
                        backend=chosen, quantize_bits=kv_bits)

        if draft:
            from reminis.speculative import Speculator, open_drafter
            drafter = open_drafter(draft, model, capacity=capacity,
                                   temperature=temperature, top_p=top_p,
                                   rng=rng, backend=chosen, pack_bits=pack_bits)
            speculator = Speculator(model, drafter, draft_tokens)
            if verbose:
                print(f"drafting {draft_tokens} tokens a round from "
                      f"{drafter.description}")

        t0 = time.time()
        logits = model.forward(tokens, cache, offset=0)
        prefill_seconds = time.time() - t0

        if (pack_bits is not None and chosen.can_pack()
                and model.store.packed == 0):
            # Bit-exact packing rearranges quantization blocks, and a float
            # model has none to rearrange. Saying so beats leaving someone to
            # wonder why a flag they passed changed nothing.
            print(
                f"Note: --pack {pack_bits} had nothing to pack. This model's "
                f"weights are stored as floats, and bit-exact packing only "
                f"rearranges existing quantization blocks.\n"
                f"      Use --pack 8 to quantize them instead."
            )

        def emit(token_id: int):
            piece = tok.decode_one(token_id)
            if on_token:
                on_token(piece)
            else:
                print(piece, end="", flush=True)

        produced = []
        t1 = time.time()
        if speculator is not None:
            produced = speculator.generate(
                logits, tokens, cache, max_tokens, temperature, top_p, rng,
                eos_id=tok.eos_id, stop_at_eos=stop_at_eos, on_token=emit,
            )
        else:
            for _ in range(max_tokens):
                token_id = _sample(logits, temperature, top_p, rng)
                if stop_at_eos and tok.eos_id is not None and token_id == tok.eos_id:
                    break
                produced.append(token_id)
                emit(token_id)
                logits = model.forward([token_id], cache, offset=cache.length)

        decode_seconds = time.time() - t1
        completion = tok.decode(produced)

        if not on_token:
            print()
        if verbose:
            rate = len(produced) / decode_seconds if decode_seconds else 0
            print(f"\n{len(tokens)} prompt tokens in {prefill_seconds:.2f}s, "
                  f"{len(produced)} generated in {decode_seconds:.2f}s "
                  f"({rate:.1f} tok/s)")
            if speculator is not None:
                # The number worth reading is tokens per pass. Acceptance
                # says how good the drafter is; passes say what that was
                # worth, since a pass is one read of the weights and a read
                # of the weights is what a token costs without all this.
                per_pass = len(produced) / speculator.passes if speculator.passes else 0
                print(f"\n{speculator.accepted}/{speculator.proposed} proposals "
                      f"accepted ({speculator.acceptance:.0%}) over "
                      f"{speculator.rounds} rounds, "
                      f"{per_pass:.2f} tokens per pass of the model")
            read = model.store.bytes_read / (1024 ** 2)
            if stream:
                print(f"Read {read:,.0f} MB from SQLite across "
                      f"{model.store.reads:,} queries, cached nothing")
            else:
                print(f"Read {read:,.0f} MB from SQLite once, then reused it")

        return {
            "prompt": text,
            "completion": completion,
            "prompt_tokens": len(tokens),
            "generated_tokens": len(produced),
            "token_ids": produced,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "bytes_read": model.store.bytes_read,
            "queries": model.store.reads,
            "backend": chosen.name,
            "packed_tensors": model.store.packed,
            "draft_accepted": speculator.accepted if speculator else 0,
            "draft_proposed": speculator.proposed if speculator else 0,
            "model_passes": speculator.passes if speculator else None,
        }
    finally:
        if speculator is not None:
            speculator.drafter.close()
        model.close()


def _apply_chat_template(prompt: str, tok, think: bool = False) -> str:
    """Wrap a prompt as a chat turn, for models that use ChatML.

    The stored template is Jinja, and rendering Jinja to run one prompt is
    more machinery than it is worth. ChatML is recognisable and it is what
    the small instruct models here use; anything else is left alone rather
    than mangled into a format the model was not trained on.

    A reasoning model needs one thing more. Its template does not stop at
    the assistant marker -- it opens a thinking channel too, and the model
    was trained expecting to find one already open:

        <|im_start|>assistant\\n<think>\\n

    Left off, the model opens the channel itself and reasons for as long as
    it likes before answering, so a run with any sane token budget shows
    working and no answer. That reads exactly like a broken forward pass
    and is not one. Closing the channel immediately -- an empty pair of
    tags -- is how these templates express "answer directly", and it is
    the default here because a one-shot completion is the case `run`
    serves. `--think` asks for the channel left open.
    """
    template = tok.chat_template or ""
    if "<|im_start|>" not in template:
        return prompt
    turn = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if "<think>" not in template:
        return turn
    return turn + ("<think>\n" if think else "<think>\n\n</think>\n\n")


def run_cli(args, on_error=None):
    """Entry point for `reminis run`, kept here so the CLI stays thin."""
    try:
        generate(
            args.input, args.prompt,
            max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p,
            seed=args.seed, stream=args.stream, chat=args.chat,
            think=getattr(args, 'think', False),
            verbose=not args.quiet, backend=args.backend,
            pack_bits=args.pack, kv_bits=args.kv_bits,
            expert_cache=0 if args.experts in (None, "all") else args.experts,
            preload=args.experts == "all",
            draft=getattr(args, "draft", None),
            draft_tokens=getattr(args, "draft_tokens", 4),
        )
    except (UnsupportedModel, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
