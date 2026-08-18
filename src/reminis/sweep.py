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

from reminis import bitmix
from reminis.db import open_read_only
from reminis.backend import select as select_backend
from reminis.errors import UnsupportedModel
from reminis.kvcache import KVCache
from reminis.model import Model

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

    `bits` is one width for the whole model, or a width per role, which is
    what a mix is. In the second case a tensor no role claims is counted at
    full precision, because that is what will happen to it: norms and biases
    are never packed at any rung.
    """
    conn = open_read_only(db_path)
    try:
        if not isinstance(bits, dict):
            total = conn.execute(
                "SELECT SUM(n_elements) FROM tensors").fetchone()[0] or 0
            return int(total * (16 if bits is None else bits + 2) / 8)
        rows = conn.execute("SELECT name, n_elements FROM tensors").fetchall()
    finally:
        conn.close()

    total_bits = 0
    for name, n_elements in rows:
        width = bits.get(bitmix.role_of(name))
        total_bits += (n_elements or 0) * (16 if width is None else width + 2)
    return int(total_bits / 8)


def _tensor_names(db_path: str) -> list:
    conn = open_read_only(db_path)
    try:
        return [name for (name,) in conn.execute("SELECT name FROM tensors")]
    finally:
        conn.close()


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


def _plan(db_path: str, prompt: str, bits_list: list, backend) -> dict:
    """The prompt, the backend, and the rung everything gets measured against.

    Pulled out because the mix sweep has to make exactly the same choice as
    the uniform one. If the two picked their reference separately, a mix
    measured against 8-bit could be compared against uniform rungs measured
    against float16, and every number in the comparison would be off by a
    different amount.

    The reference is the most accurate rung that fits: full precision when
    there is room for it, and otherwise the widest requested packing that
    fits, which the report then says out loud rather than quietly comparing
    against something else.
    """
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

    memory_budget = probe_budget = None
    try:
        probe_budget = chosen.memory_budget()
    except Exception:
        pass
    memory_budget = int(probe_budget * FIT_SHARE) if probe_budget else None

    ladder = [None] + sorted(set(bits_list), reverse=True)
    reference_bits = None
    if memory_budget:
        for candidate in ladder:
            if _predicted_bytes(db_path, candidate) <= memory_budget:
                reference_bits = candidate
                break
        else:
            raise UnsupportedModel(
                f"Even at {min(bits_list)} bits this model needs about "
                f"{_predicted_bytes(db_path, min(bits_list)) / 1e9:.1f} GB "
                f"resident, and only {memory_budget / 1e9:.1f} GB is usable on "
                "this machine. Sweep a smaller model, or a narrower --bits list."
            )

    return {
        "tokens": tokens,
        "backend": chosen,
        "reference_bits": reference_bits,
        "reference_label": "f16" if reference_bits is None
                           else f"{reference_bits}*",
        "memory_budget": memory_budget,
        "did_not_fit": [b for b in sorted(set(bits_list), reverse=True)
                        if reference_bits is not None and b > reference_bits],
    }


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
    result, _ = _sweep(db_path, bits_list, prompt, backend, kv_bits, verbose)
    return result


def _sweep(db_path, bits_list, prompt, backend, kv_bits, verbose, plan=None):
    """The uniform sweep, handing back the reference logits it computed.

    The mix sweep needs those logits to score its own probes against. Loading
    the reference a second time to get them is a whole model load, which on a
    7B model is minutes -- and worse, it would be a *second* reference, so a
    difference between the two runs would show up as a difference between the
    uniform table and the mix rather than as what it is.
    """
    prompt = prompt or DEFAULT_PROMPT
    plan = plan or _plan(db_path, prompt, bits_list, backend)
    tokens, chosen = plan["tokens"], plan["backend"]
    reference_bits, budget = plan["reference_bits"], plan["memory_budget"]
    ref_label, skipped_wide = plan["reference_label"], plan["did_not_fit"]

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
    return result, reference


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


# How much of the reference's top-1 agreement a group is allowed to cost
# before its width is rejected. A judgement call, so it is a parameter with a
# stated default rather than a constant buried in the search.
DEFAULT_BUDGET = 0.99


def _source_widths(db_path: str) -> dict:
    """How many packable tensors the file holds at each stored type.

    A GGUF named `Q4_K_M` is already a mix -- that is exactly what the `_M`
    means, some tensors at Q6_K and the rest at Q4_K -- so a model converted
    from one carries llama.cpp's recipe in its dtypes and can be measured
    without any of this machinery. That makes it the honest baseline: not
    "is a mix better than a uniform width", which is a question llama.cpp
    settled years ago, but "is a mix derived for this model better than the
    one recipe shipped for every model".

    More than one type here means the source was mixed. One type means it was
    not, and there is no baseline to read off the file.
    """
    conn = open_read_only(db_path)
    try:
        rows = conn.execute("SELECT name, dtype FROM tensors").fetchall()
    finally:
        conn.close()
    counts = {}
    for name, dtype in rows:
        if bitmix.role_of(name) is not None:
            counts[dtype] = counts.get(dtype, 0) + 1
    return counts


def _mix_of(roles: list, base, override=None, bits=None) -> dict:
    """Every role at `base`, except one held at a different width."""
    mix = dict.fromkeys(roles, base)
    if override is not None:
        mix[override] = bits
    return mix


def mix_sweep(
    db_path: str,
    bits_list: list,
    prompt: str | None = None,
    backend: str | None = None,
    kv_bits: int | None = None,
    budget: float = DEFAULT_BUDGET,
    verbose: bool = True,
) -> dict:
    """Derive a per-group width map for this model, and check it beats uniform.

    The uniform sweep asks "what does 4-bit cost for this model?" and answers
    with one number. That number is a compromise: it is set by whichever
    tensors tolerate 4 bits least, and every other tensor in the model pays
    for them. Which tensors those are is not the same from model to model,
    and it is measurable, so it is measured here rather than assumed.

    The procedure, in full, because the result means nothing without it:

      1. Run the uniform rungs, to have something to beat.
      2. Hold every group at the widest rung that fits, and take one group
         at a time down to each narrower width, measuring agreement.
      3. Keep, for each group, the narrowest width that stayed inside the
         agreement budget.
      4. Measure that mix end to end, because per-group errors do not have
         to add up the way step 3 assumes they do -- step 3 proposes and
         this step is what decides.
      5. Say plainly whether the result beats the uniform rungs on either
         axis, or does not.

    Step 2 walks each group from wide to narrow and stops at the first width
    that misses the budget. That assumes a group does not get *better* as it
    gets narrower, which is not a theorem; widths skipped that way are
    reported as not attempted rather than as failures.
    """
    prompt = prompt or DEFAULT_PROMPT
    plan = _plan(db_path, prompt, bits_list, backend)
    tokens = plan["tokens"]
    reference_bits, memory_budget = plan["reference_bits"], plan["memory_budget"]

    widths = sorted(set(bits_list), reverse=True)
    # Everything starts at the widest rung that both fits and was asked for.
    # That is the reference itself where the machine could not hold float16,
    # and then the mix is being asked to buy memory back from a rung already
    # known to be adequate.
    usable = [b for b in widths
              if reference_bits is None or b <= reference_bits]
    if len(usable) < 2:
        raise UnsupportedModel(
            "A mix needs at least two widths to choose between; only "
            f"{usable or 'none'} of --bits fits on this machine."
        )
    base, candidates = usable[0], usable[1:]

    roles = bitmix.roles_in(_tensor_names(db_path))
    if not roles:
        raise UnsupportedModel(
            "None of this model's tensors are ones reminis packs, so there "
            "is no width to choose per group."
        )

    if verbose:
        print(f"Deriving a per-group mix for {db_path}")
        print(f"{len(tokens)} prompt tokens, groups: {', '.join(roles)}")
        print(f"Holding everything at {base}-bit and taking one group at a "
              f"time down to {', '.join(f'{b}' for b in candidates)}.")
        print(f"Keeping the narrowest width that holds {100 * budget:.0f}% of "
              f"what uniform {base}-bit agrees on.\n")

    uniform, reference = _sweep(db_path, bits_list, prompt, backend, kv_bits,
                                verbose, plan=plan)
    ref_bytes = uniform["rows"][0]["bytes"]

    # The budget is a share of what the base rung itself achieves, not of the
    # reference. Measuring against the reference conflates two questions: what
    # quantizing this model costs at all, and what narrowing *this group*
    # costs on top of that. Only the second is what the search is choosing on,
    # and on a model where uniform 8-bit already gives up five points a budget
    # against the reference is unreachable for every group at every width --
    # the search would return the base width and call it a derived mix.
    base_top1 = _base_agreement(uniform, base)
    threshold = base_top1 * budget
    if verbose and base_top1 < 1.0:
        print(f"\nUniform {base}-bit agrees with the reference {100*base_top1:.1f}%"
              f" of the time, so a group\nkeeps a narrower width if it holds "
              f"{100*threshold:.1f}% -- {100*budget:.0f}% of that.")

    # The baseline worth beating, when the file carries one. `compact` keeps
    # every block in the width it was stored at wherever that block has an
    # exact affine form, so this measures llama.cpp's own mix rather than an
    # approximation of it -- except for the types that have none, which are
    # re-quantized to the nearest width and are counted apart below.
    source = _source_widths(db_path)
    native = None
    if len(source) > 1:
        if verbose:
            label = ", ".join(f"{d}x{n}" for d, n in sorted(source.items()))
            print(f"\nMeasuring the source's own mix ({label})", flush=True)
        try:
            logits, resident, elapsed = _logits_for(
                db_path, tokens, "compact", backend, kv_bits)
        except Exception as exc:
            native = {"widths": source, "skipped": str(exc)}
        else:
            native = {"widths": source, "bytes": resident,
                      "seconds": elapsed, **_compare(reference, logits)}

    probes, chosen = [], dict.fromkeys(roles, base)
    for role in roles:
        for width in candidates:
            trial = _mix_of(roles, base, role, width)
            predicted = _predicted_bytes(db_path, trial)
            if memory_budget and predicted > memory_budget:
                probes.append({"role": role, "bits": width,
                               "skipped": "would not fit resident"})
                break
            if verbose:
                print(f"Probing {role} at {width}-bit", flush=True)
            try:
                logits, resident, _ = _logits_for(
                    db_path, tokens, trial, backend, kv_bits)
            except Exception as exc:  # a probe that cannot run is data
                probes.append({"role": role, "bits": width,
                               "skipped": str(exc)})
                break
            stats = _compare(reference, logits)
            probes.append({"role": role, "bits": width, "bytes": resident,
                           **stats})
            if stats["top1"] < threshold:
                # This group cannot take this width, and by the assumption
                # stated above it will not take a narrower one either. The
                # widths below are recorded as skipped so the table does not
                # read as though they had been tried and passed.
                for narrower in candidates[candidates.index(width) + 1:]:
                    probes.append({"role": role, "bits": narrower,
                                   "skipped": "not attempted, "
                                              f"{role} already failed at {width}"})
                break
            chosen[role] = width

    # Measured rather than assembled from the probes, because a group's cost
    # measured on its own does not have to be its cost alongside the others.
    #
    # When no group could take a narrower width this re-measures the uniform
    # base rung, which looks like waste and is worth the load: the two numbers
    # come from separate model loads of the same configuration, so they have
    # to agree, and if they ever do not then something here is not
    # deterministic and every other row is suspect.
    if verbose:
        print(f"\nMeasuring the derived mix: {bitmix.describe(chosen)}",
              flush=True)
    derived = None
    try:
        logits, resident, elapsed = _logits_for(
            db_path, tokens, chosen, backend, kv_bits)
    except Exception as exc:
        derived = {"mix": chosen, "skipped": str(exc)}
    else:
        derived = {
            "mix": chosen, "bytes": resident,
            "share": resident / ref_bytes if ref_bytes else 0.0,
            "seconds": elapsed, **_compare(reference, logits),
        }

    result = {
        "model": db_path,
        "prompt_tokens": len(tokens),
        "reference_bits": reference_bits,
        "reference_bytes": ref_bytes,
        "base_bits": base,
        "budget": budget,
        "base_top1": base_top1,
        "threshold": threshold,
        "roles": roles,
        "probes": probes,
        "derived": derived,
        "source_mix": native,
        "uniform": uniform,
        "verdict": _verdict(derived, uniform),
    }
    if verbose:
        _print_mix(result)
    return result


def _base_agreement(uniform: dict, base: int) -> float:
    """What the all-at-`base` rung agrees with the reference on.

    It is the reference itself on a model too big to hold at float16, in
    which case it agrees with itself and the budget is read straight.
    """
    for row in uniform["rows"]:
        if row.get("bits") in (base, f"{base}*") and "skipped" not in row:
            return row["top1"]
    return 1.0


def _verdict(derived: dict, uniform: dict) -> dict:
    """Whether the derived mix is actually worth having.

    Two ways for it to be, and one way for it not to be. It wins on memory if
    no uniform rung matches its agreement for fewer bytes; it wins on
    agreement if no uniform rung of its size or smaller agrees as often. If
    some uniform rung is at least as good on both axes then the mix is
    dominated and the honest report is that this model did not need one.
    """
    if "skipped" in derived:
        return {"verdict": "not measured", "reason": derived["skipped"]}

    rungs = [r for r in uniform["rows"]
             if "skipped" not in r and isinstance(r["bits"], int)]
    if not rungs:
        return {"verdict": "no comparison",
                "reason": "no uniform rung ran to compare against"}

    mix_bytes, mix_top1 = derived["bytes"], derived["top1"]
    as_good = [r for r in rungs if r["top1"] >= mix_top1]
    as_small = [r for r in rungs if r["bytes"] <= mix_bytes]
    dominated = [r for r in rungs
                 if r["bytes"] <= mix_bytes and r["top1"] >= mix_top1]

    cheapest_as_good = min(as_good, key=lambda r: r["bytes"]) if as_good else None
    best_as_small = max(as_small, key=lambda r: r["top1"]) if as_small else None

    return {
        "verdict": "dominated by uniform" if dominated else "beats uniform",
        "dominated_by": [r["bits"] for r in dominated],
        "bytes_saved": (cheapest_as_good["bytes"] - mix_bytes
                        if cheapest_as_good else None),
        "matched_rung": cheapest_as_good["bits"] if cheapest_as_good else None,
        "top1_gained": (mix_top1 - best_as_small["top1"]
                        if best_as_small else None),
        "undercut_rung": best_as_small["bits"] if best_as_small else None,
    }


def _print_mix(result: dict):
    threshold = result["threshold"]
    print("\n" + "=" * 74)
    print(f"{'group':>8} {'bits':>6} {'resident':>10} {'top-1':>8} "
          f"{'top-5':>8} {'KL':>10}")
    print("-" * 74)
    for probe in result["probes"]:
        head = f"{probe['role']:>8} {probe['bits']:>6}"
        if "skipped" in probe:
            print(f"{head}   {probe['skipped']}")
            continue
        mark = " " if probe["top1"] >= threshold else "x"
        print(f"{head} {probe['bytes']/1e6:7.0f} MB {100*probe['top1']:7.1f}%"
              f"{mark}{100*probe['top5']:7.1f}% {probe['kl']:10.4f}")
    print("=" * 74)
    print(f"x = below the {100*threshold:.1f}% budget, so this group keeps "
          f"the wider width.\nEach row holds every other group at "
          f"{result['base_bits']}-bit.")

    derived = result["derived"]
    print(f"\nDerived mix: {bitmix.describe(derived['mix'])}")
    if "skipped" in derived:
        print(f"  could not be measured: {derived['skipped']}")
        return
    print(f"  {derived['bytes']/1e6:.0f} MB resident, "
          f"{100*derived['share']:.1f}% of the reference, "
          f"top-1 {100*derived['top1']:.1f}%, top-5 "
          f"{100*derived['top5']:.1f}%, KL {derived['kl']:.4f}")

    print("\nAgainst the uniform rungs, on the same reference and prompt:")
    for row in result["uniform"]["rows"]:
        if "skipped" in row:
            continue
        print(f"  {str(row['bits']):>4}  {row['bytes']/1e6:7.0f} MB  "
              f"top-1 {100*row['top1']:6.1f}%")
    native = result.get("source_mix")
    if native and "skipped" not in native:
        print(f"  {'gguf':>4}  {native['bytes']/1e6:7.0f} MB  "
              f"top-1 {100*native['top1']:6.1f}%   "
              f"the source file's own mix, as llama.cpp chose it")
    print(f"  {'mix':>4}  {derived['bytes']/1e6:7.0f} MB  "
          f"top-1 {100*derived['top1']:6.1f}%")

    native = result.get("source_mix")
    if native and "skipped" not in native:
        _print_against_source(derived, native)

    verdict = result["verdict"]
    print()
    if verdict["verdict"] == "dominated by uniform":
        rungs = ", ".join(f"{b}-bit" for b in verdict["dominated_by"])
        print(f"The mix is dominated by {rungs}: at least as much agreement "
              f"for at most as many\nbytes. On this model the per-group "
              f"search did not pay, and the honest conclusion\nis to use a "
              f"uniform width.")
        return
    print("The mix is not dominated by any uniform rung.")
    if verdict["bytes_saved"] and verdict["bytes_saved"] > 0:
        print(f"  {verdict['bytes_saved']/1e6:.0f} MB less than "
              f"{verdict['matched_rung']}-bit, the cheapest uniform rung that "
              f"agrees as often.")
    if verdict["top1_gained"] and verdict["top1_gained"] > 0:
        print(f"  {100*verdict['top1_gained']:.1f} points more top-1 than "
              f"{verdict['undercut_rung']}-bit, which is no larger.")


def _print_against_source(derived: dict, native: dict):
    """The derived mix beside the recipe the file arrived with.

    Both are mixes, so neither can claim the idea. What is being compared is
    where the widths came from: measured on this model, or chosen once for
    every model. A tie is a real answer and is printed as one -- it means the
    fixed recipe already suited this model, which is not nothing.
    """
    bytes_delta = native["bytes"] - derived["bytes"]
    top1_delta = derived["top1"] - native["top1"]
    print(f"\nAgainst the source file's own mix: "
          f"{abs(bytes_delta)/1e6:.0f} MB "
          f"{'less' if bytes_delta > 0 else 'more'}, "
          f"{abs(100*top1_delta):.1f} points "
          f"{'more' if top1_delta > 0 else 'less'} top-1.")
    if bytes_delta >= 0 and top1_delta >= 0:
        print("Better or equal on both, so deriving the mix beat inheriting it.")
    elif bytes_delta < 0 and top1_delta < 0:
        print("Worse on both. The fixed recipe won on this model, and the "
              "derived mix is not\nworth the measurement time it cost.")
    else:
        print("Better on one axis and worse on the other, which is a trade "
              "rather than a win.")
