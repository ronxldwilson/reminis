"""Run one model at several precisions and report what each rung costs.

Picking a quantization is usually done by reading someone else's table and
hoping it transfers. It does not always: how much a model tolerates depends on
the model, and the only way to know is to run the thing and look.

That is awkward with a directory of converted files -- every rung is another
full copy on disk, and comparing them means keeping all of them. Here the
weights are rows, the packed form is derived from those rows on the way to the
GPU, and the original bytes never move, so a rung costs nothing once measured
and the exact source stays byte-intact beside the results.

What is measured, against the same model at full precision on the same prompt:

  resident bytes   what the weights occupy once packed, which is the number
                   that decides whether the model fits
  top-1 agreement  how often the packed model's best next token is the one
                   the full-precision model would have picked
  top-5 overlap    how much of the candidate set survives -- a model can keep
                   its top-1 while shuffling everything behind it
  KL divergence    the whole distribution, in nats, which catches damage the
                   ranking measures miss

Agreement is measured over the prompt's own positions rather than over
generated text, because sampling makes two runs diverge for reasons that have
nothing to do with precision.
"""

import time

import numpy as np

from reminis.db import open_read_only
from reminis.backend import select as select_backend
from reminis.infer import KVCache, Model, UnsupportedModel

# Enough tokens that the agreement percentages mean something, short enough
# that a rung is seconds rather than minutes.
DEFAULT_PROMPT = (
    "The capital of France is Paris. The capital of Japan is Tokyo. "
    "Machine learning models are trained on large datasets to predict "
    "the next token in a sequence. In 1969, humans first walked on the "
    "surface of the Moon, an achievement that required enormous advances "
    "in computing, materials science, and propulsion."
)


# A rung is only attempted if its predicted resident size leaves this much of
# the working set free. Going over does not fail -- unified memory just starts
# compressing and swapping -- so the run would appear to hang rather than
# stop, which is the worst way to find out a model does not fit.
FIT_SHARE = 0.8


def _predicted_bytes(db_path: str, bits) -> int:
    """Roughly what the weights will occupy resident, at a given width.

    Counted from element counts rather than file size because the file size is
    the same at every rung; what changes is what the weights expand or pack to
    on the way to the backend. Group scales add about two bits per weight at
    the default group size, which is included.
    """
    conn = open_read_only(db_path)
    try:
        total = conn.execute("SELECT SUM(n_elements) FROM tensors").fetchone()[0] or 0
    finally:
        conn.close()
    per_weight = 16 if bits is None else bits + 2
    return int(total * per_weight / 8)


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _sizeof(value) -> int:
    """Bytes held by one cached weight, packed or plain."""
    q = getattr(value, "q", None)
    if q is not None:
        return sum(int(p.nbytes) for p in
                   (q, value.scales, value.biases) if p is not None)
    nbytes = getattr(value, "nbytes", None)
    return int(nbytes) if nbytes is not None else 0


def _resident_bytes(model) -> int:
    """What the cached weights actually occupy, packed or not.

    Counted from the objects handed back rather than from the file, because
    that is the point of the exercise: the file is the same size at every rung
    and only the resident form changes.

    Both caches have to be counted. At full precision the model fuses Q/K/V and
    gate/up into single matrices held in its own cache and drops the originals
    from the store's; packed weights cannot be stacked, so they stay put.
    Counting only the store would therefore undercount exactly the reference
    the other rungs are measured against, and make packing look like a loss.
    """
    total = sum(_sizeof(v) for v in model.store._cache.values())
    for weight, bias in model._fused_cache.values():
        total += _sizeof(weight)
        if bias is not None:
            total += _sizeof(bias)
    return total


def _logits_for(db_path, tokens, bits, backend_name, kv_bits=None):
    """Full-vocabulary logits at every prompt position, for one precision.

    One batched prefill, not one call per token. Every position's logits fall
    out of the same pass, and a single matrix product over the whole prompt is
    far cheaper than a few dozen matrix-vector products over one token each.
    """
    chosen = select_backend("inference", backend_name)
    model = Model(db_path, backend=chosen, pack_bits=bits)
    try:
        cache = KVCache(model.cfg.n_layers, capacity=len(tokens),
                        backend=model.backend, quantize_bits=kv_bits)
        started = time.perf_counter()
        logits = model.forward(tokens, cache, 0, all_positions=True)
        elapsed = time.perf_counter() - started
        return np.asarray(logits, dtype=np.float32), _resident_bytes(model), elapsed
    finally:
        model.close()


def _compare(reference, candidate):
    """Ranking and distribution agreement between two sets of logits."""
    if reference.ndim == 1:
        reference = reference[None, :]
        candidate = candidate[None, :]

    ref_top1 = reference.argmax(axis=-1)
    cand_top1 = candidate.argmax(axis=-1)
    top1 = float((ref_top1 == cand_top1).mean())

    k = min(5, reference.shape[-1])
    ref_top5 = np.argpartition(-reference, k - 1, axis=-1)[:, :k]
    cand_top5 = np.argpartition(-candidate, k - 1, axis=-1)[:, :k]
    overlap = np.mean([
        len(set(a.tolist()) & set(b.tolist())) / k
        for a, b in zip(ref_top5, cand_top5)
    ])

    p = _softmax(reference)
    q = _softmax(candidate)
    kl = float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=-1)))

    return {"top1": top1, "top5": float(overlap), "kl": kl}


def sweep(
    db_path: str,
    bits_list: list,
    prompt: str | None = None,
    backend: str | None = None,
    kv_bits: int | None = None,
    verbose: bool = True,
) -> dict:
    """Measure a model at each precision in `bits_list` against full precision.

    Returns a dict with one entry per rung. Rungs that the backend cannot pack
    are reported as skipped rather than silently dropped, so a table with a
    missing row always says why.
    """
    prompt = prompt or DEFAULT_PROMPT

    chosen = select_backend("inference", backend)
    probe = Model(db_path, backend=chosen)
    try:
        tokens = probe.tokenizer.encode(prompt)
        can_pack = probe.backend.can_pack()
    finally:
        probe.close()

    if not can_pack:
        raise UnsupportedModel(
            f"The {chosen} backend cannot hold packed weights, so there is "
            "nothing to sweep. Quantized inference needs the MLX backend."
        )

    # Everything is measured against the most accurate rung that fits. That is
    # full precision when there is room for it; on a model too big for the
    # machine at float16 it is the widest requested packing that fits, and the
    # table says so rather than quietly comparing against something else.
    budget = probe_budget = None
    try:
        probe_budget = chosen.memory_budget()
    except Exception:
        pass
    budget = int(probe_budget * FIT_SHARE) if probe_budget else None

    ladder = [None] + sorted(set(bits_list), reverse=True)
    reference_bits = None
    if budget:
        for candidate in ladder:
            if _predicted_bytes(db_path, candidate) <= budget:
                reference_bits = candidate
                break
        else:
            raise UnsupportedModel(
                f"Even at {min(bits_list)} bits this model needs about "
                f"{_predicted_bytes(db_path, min(bits_list)) / 1e9:.1f} GB "
                f"resident, and only {budget / 1e9:.1f} GB is usable on this "
                "machine. Sweep a smaller model, or a narrower --bits list."
            )

    ref_label = "f16" if reference_bits is None else f"{reference_bits}*"
    skipped_wide = [b for b in sorted(set(bits_list), reverse=True)
                    if reference_bits is not None and b > reference_bits]

    if verbose:
        print(f"Sweeping {db_path}")
        print(f"{len(tokens)} prompt tokens, "
              f"{type(chosen).__name__.replace('Backend', '').lower()} backend")
        if reference_bits is not None:
            print(f"\nFull precision would need about "
                  f"{_predicted_bytes(db_path, None) / 1e9:.1f} GB resident, over "
                  f"the {budget / 1e9:.1f} GB usable here.\n"
                  f"Measuring against {reference_bits}-bit instead, the widest "
                  f"rung that fits. Percentages are relative to it, not to f16.")
        print(f"\nMeasuring {ref_label} (the reference)", flush=True)

    reference, ref_bytes, ref_time = _logits_for(
        db_path, tokens, reference_bits, backend, kv_bits)

    rows = [{
        "bits": ref_label, "bytes": ref_bytes, "share": 1.0, "seconds": ref_time,
        "top1": 1.0, "top5": 1.0, "kl": 0.0,
    }]

    for bits in bits_list:
        if reference_bits is not None and bits >= reference_bits:
            continue  # already measured, or too wide to fit
        if verbose:
            print(f"Measuring {bits}-bit", flush=True)
        try:
            logits, resident, elapsed = _logits_for(
                db_path, tokens, bits, backend, kv_bits)
        except Exception as exc:  # a rung that cannot run is data, not a crash
            rows.append({"bits": bits, "skipped": str(exc)})
            continue
        stats = _compare(reference, logits)
        rows.append({
            "bits": bits, "bytes": resident,
            "share": resident / ref_bytes if ref_bytes else 0.0,
            "seconds": elapsed, **stats,
        })

    result = {
        "model": db_path, "prompt_tokens": len(tokens),
        "reference_bits": reference_bits, "rows": rows,
        "did_not_fit": skipped_wide,
    }
    if verbose:
        _print(result)
    return result


def _print(result: dict):
    ref = result.get("reference_bits")
    against = "of f16" if ref is None else f"of {ref}b"
    print("\n" + "=" * 74)
    print(f"{'bits':>6} {'resident':>10} {against:>8} {'top-1':>8} "
          f"{'top-5':>8} {'KL':>10} {'forward':>9}")
    print("-" * 74)
    for row in result["rows"]:
        if "skipped" in row:
            print(f"{str(row['bits']):>6}   skipped: {row['skipped']}")
            continue
        print(f"{str(row['bits']):>6} {row['bytes']/1e6:7.0f} MB "
              f"{100*row['share']:7.1f}% {100*row['top1']:7.1f}% "
              f"{100*row['top5']:7.1f}% {row['kl']:10.4f} "
              f"{row['seconds']:8.2f}s")
    print("=" * 74)

    if result.get("did_not_fit"):
        widths = ", ".join(f"{b}-bit" for b in result["did_not_fit"])
        print(f"\nNot attempted, would not fit resident: f16, {widths}.")
    if result.get("reference_bits") is not None:
        print(f"* the reference. It is itself quantized, so agreement here is "
              f"agreement with {result['reference_bits']}-bit, not with the "
              f"original weights.")

    ref_label = result["rows"][0]["bits"]
    usable = [r for r in result["rows"]
              if "skipped" not in r and r["bits"] != ref_label]
    if not usable:
        return
    # The cheapest rung that still picks the same token essentially always.
    # 99% is a judgement call, stated rather than hidden, and the whole table
    # is printed so a different threshold can be applied by eye.
    good = [r for r in usable if r["top1"] >= 0.99]
    if good:
        best = min(good, key=lambda r: r["bytes"])
        print(f"\nCheapest rung holding top-1 agreement at 99% or better: "
              f"{best['bits']}-bit, {100*best['share']:.0f}% of full precision.")
    else:
        print("\nNo rung held top-1 agreement at 99%. This model is more "
              "precision-sensitive than the usual defaults assume.")
