"""Guess several tokens, then check them all in one pass.

Decoding is memory-bound at batch size one. Every weight in the model is
read to produce a single token, so the rate is the model's size divided by
the machine's memory bandwidth and almost nothing else -- measured here on
a 10.2 GB model and an Apple M5, 102 ms a token against an implied
100 GB/s, which is the bus rather than the arithmetic.

The consequence is that reading the weights *once* to check five tokens
costs about what reading them once to produce one costs. So: something
cheap proposes a few tokens, the real model runs them as a single batch,
and the proposals that agree with what it would have said anyway are kept
for free. What makes this worth doing rather than merely clever is that
the output is not an approximation -- the accept/reject rule below returns
exactly the distribution the target model would have sampled from on its
own, so a run with a draft and a run without it are the same model, at
different speeds.

Two proposers are here, and they differ in what they cost rather than in
how they are treated:

  ``ModelDrafter``  a second, smaller model. It has to share a tokenizer
      with the target for the ids to mean the same thing, which is checked
      at load rather than trusted -- reminis keeps the tokenizer *in* the
      database, so the check is a comparison of two blobs rather than an
      article of faith about two downloads.
  ``NgramDrafter``  no model at all. It looks for the last few tokens
      somewhere earlier in the context and proposes whatever followed them
      then. Free, needs no memory, and helps exactly where output repeats
      input -- summarising, editing code, answering from a quoted document.

What has to be undone
---------------------

A rejected proposal has already been through the model, so the pass has to
be undone. For attention that is `KVCache.rollback`, which lowers a counter
and is free. For the recurrent layers in a hybrid model like qwen35 it is
not possible at all: three quarters of the layers fold each token into a
hidden state that keeps no per-token record to truncate.

So this does not try. `Model.snapshot_state` returns the state from before
the pass, and on a partial acceptance the accepted tokens go back into
`pending` to be re-read at the start of the *next* verification pass --
which has to happen regardless. The re-read costs arithmetic and no extra
trip to memory, which on a machine that is bandwidth-bound is close to
free, and it is why nothing here has to hold a state per speculated
position. `pending` is capped so that a long run of rejections cannot let
the batch grow without bound.
"""

import numpy as np

from reminis.errors import UnsupportedModel
from reminis.kvcache import KVCache

__all__ = ["NgramDrafter", "ModelDrafter", "Speculator", "open_drafter"]


# ------------------------------------------------------------------ drafters


class Drafter:
    """Something that proposes the next few tokens."""

    #: Shown in the run header.
    description = "a drafter"

    def propose(self, sequence: list[int], k: int):
        """Propose up to `k` continuations of `sequence`.

        Returns (tokens, distributions). `distributions` is either a list
        of full-vocabulary probability vectors -- one per proposed token,
        the distribution the proposal was drawn from -- or None, meaning
        the proposal was made with certainty and should be treated as a
        point mass. The accept/reject rule below needs one or the other to
        stay exact.
        """
        raise NotImplementedError

    def close(self):
        pass


class NgramDrafter(Drafter):
    """Propose from the context itself, with no second model.

    Where the answer repeats the question -- summarising a document,
    editing a file, quoting back a definition -- the next few tokens have
    usually already been seen. Matching the longest recent suffix against
    everything before it and proposing what followed costs a dictionary
    lookup and no memory at all.

    Longest match first, most recent occurrence first: a short match is
    likelier to be a coincidence, and the nearest occurrence is likelier to
    be the passage actually being echoed.
    """

    description = "n-gram lookup (no draft model)"

    def __init__(self, max_ngram: int = 4, min_ngram: int = 2):
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram

    def propose(self, sequence: list[int], k: int):
        n = len(sequence)
        for size in range(min(self.max_ngram, n - 1), self.min_ngram - 1, -1):
            needle = sequence[n - size:]
            # Nearest occurrence first, and never the suffix itself.
            for start in range(n - size - 1, -1, -1):
                if sequence[start:start + size] == needle:
                    found = sequence[start + size:start + size + k]
                    if found:
                        return list(found), None
                    break
        return [], None


class ModelDrafter(Drafter):
    """A second, smaller model in the same tokenizer.

    It runs the ordinary decode loop `k` times and then rolls itself back,
    so its cache and state hold only what the target has actually accepted.
    Rolling back rather than committing is what keeps the two in step
    without the caller having to track two lengths.
    """

    def __init__(self, model, capacity: int, temperature: float, top_p: float,
                 rng, label: str = ""):
        self.model = model
        self.cache = KVCache(model.cfg.n_layers, capacity=capacity,
                             backend=model.backend)
        self.temperature = temperature
        self.top_p = top_p
        self.rng = rng
        self.description = label or model.meta.get("general.name", "draft model")
        self._fed = 0
        self._logits = None

    def propose(self, sequence: list[int], k: int):
        from reminis.infer import _probs

        fresh = sequence[self._fed:]
        if fresh:
            self._logits = self.model.forward(fresh, self.cache,
                                              offset=self.cache.length)
            self._fed = len(sequence)

        # Everything from here is provisional, so where it began is recorded
        # and put back before returning.
        snapshot = self.model.snapshot_state()
        base = self.cache.length

        logits = self._logits
        tokens, distributions = [], []
        for i in range(k):
            if self.temperature <= 0:
                token, dist = int(np.argmax(logits)), None
            else:
                dist = _probs(logits, self.temperature, self.top_p)
                token = int(self.rng.choice(len(dist), p=dist))
            tokens.append(token)
            distributions.append(dist)
            if i + 1 < k:
                logits = self.model.forward([token], self.cache,
                                            offset=self.cache.length)

        self.model.restore_state(snapshot)
        self.cache.rollback(base)
        return tokens, (None if self.temperature <= 0 else distributions)

    def close(self):
        self.model.close()


# ------------------------------------------------------------- compatibility


def _tokenizer_fingerprint(meta: dict) -> tuple:
    """What has to match for two models' token ids to mean the same thing.

    The vocabulary, hashed, and the family that reads it. Deliberately not
    the special-token ids: only the target's end-of-text is consulted, and
    two files from the same family often disagree about whether to record
    a beginning-of-text at all -- the Qwen3.5-4B build here has no
    `bos_token_id` row where the 27B does, which changes the meaning of no
    id and would be a false refusal.
    """
    import hashlib

    tokens = (meta.get("tokenizer.ggml.tokens") or "").encode()
    return (
        meta.get("tokenizer.ggml.model"),
        meta.get("tokenizer.ggml.pre"),
        hashlib.sha256(tokens).hexdigest()[:16],
    )


def check_compatible(target, draft) -> None:
    """Refuse a draft model whose ids do not mean the same as the target's.

    A mismatched pair does not fail: it produces text. The draft proposes
    id 5012 meaning one word, the target reads id 5012 as another, rejects
    nearly everything, and the run is correct but slower than not having
    bothered -- or, with a vocabulary that only partly overlaps, quietly
    wrong. reminis stores the tokenizer in the database next to the
    weights, so the pair can be compared instead of assumed.
    """
    want, got = _tokenizer_fingerprint(target.meta), _tokenizer_fingerprint(draft.meta)
    if want != got:
        raise UnsupportedModel(
            f"The draft model's tokenizer is not the target's, so their "
            f"token ids do not mean the same thing.\n"
            f"  target: {target.meta.get('general.name', '?')}  {want}\n"
            f"  draft:  {draft.meta.get('general.name', '?')}  {got}\n"
            f"A draft has to come from the same family as the model it "
            f"drafts for."
        )


def open_drafter(spec: str, target, capacity: int, temperature: float,
                 top_p: float, rng, backend=None, pack_bits=None):
    """Build a drafter from what `--draft` was given.

    ``ngram``            the context-lookup drafter, no model
    ``path/to.db``       a second model database
    ``registry.db#name`` one model out of a registry, materialised first
    """
    if spec in ("ngram", "n-gram", "lookup"):
        return NgramDrafter()

    from reminis.model import Model

    path, _, name = spec.partition("#")
    if name:
        return _from_registry(path, name, target, capacity, temperature,
                              top_p, rng, backend, pack_bits)

    draft = Model(path, backend=backend, pack_bits=pack_bits)
    try:
        check_compatible(target, draft)
    except Exception:
        draft.close()
        raise
    return ModelDrafter(draft, capacity, temperature, top_p, rng)


def _from_registry(path, name, target, capacity, temperature, top_p, rng,
                   backend, pack_bits):
    """A draft that travels in the same file as the model it drafts for.

    A registry stores a derived model as a delta against its parent, so a
    quantized copy of the target -- which `reminis quantize` already
    produces and is a perfectly good draft -- costs a fraction of a second
    file. What it does not do is hand out a database: the tensors are
    reassembled on the way out, so this writes the draft to a scratch file
    and opens that. It is the price of the deltas, paid once per run.
    """
    import tempfile
    from pathlib import Path

    from reminis.model import Model
    from reminis.registry import Registry

    scratch = tempfile.TemporaryDirectory(prefix="reminis-draft-")
    with Registry(path, create=False) as reg:
        if name not in reg:
            raise UnsupportedModel(
                f"'{name}' is not in {path}. Run `reminis registry ls "
                f"{path}` to see what is."
            )
        db = reg.materialize(name, str(Path(scratch.name) / "draft.db"),
                             verbose=False)
        draft = Model(db, backend=backend, pack_bits=pack_bits)

    try:
        check_compatible(target, draft)
    except Exception:
        draft.close()
        scratch.cleanup()
        raise
    drafter = ModelDrafter(draft, capacity, temperature, top_p, rng,
                           label=f"{name} (from {Path(path).name})")
    close = drafter.close

    def close_and_clean():
        close()
        scratch.cleanup()

    drafter.close = close_and_clean
    return drafter


# ------------------------------------------------------------- the main loop


class Speculator:
    """Runs the target model against a drafter's proposals."""

    def __init__(self, model, drafter: Drafter, k: int = 4):
        if k < 1:
            raise ValueError("--draft-tokens must be at least 1")
        self.model = model
        self.drafter = drafter
        self.k = k
        self.proposed = 0
        self.accepted = 0
        self.rounds = 0
        self.passes = 0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    def generate(self, logits, prompt: list[int], cache: KVCache,
                 max_tokens: int, temperature: float, top_p: float, rng,
                 eos_id: int | None = None, stop_at_eos: bool = True,
                 on_token=None):
        """Produce tokens until `max_tokens` or the end of text.

        `logits` is the prefill's, so the caller keeps the prompt pass and
        its timing where they were. After that every round is: propose k,
        verify `pending + k` in a single pass, keep the agreeing prefix,
        and sample one more token from the position where the agreement
        stopped -- which is a token the target produced itself, so a round
        always makes progress even when every proposal is rejected.
        """
        self.passes += 1  # the prefill, which the caller has already run
        first = self._draw(logits, temperature, top_p, rng)

        # `pending` holds tokens that are accepted but that the target has
        # not read yet -- always at least the one whose successor is being
        # predicted. Everything in it is verified text; nothing here is
        # provisional.
        pending = [first]
        produced = []

        def keep(tokens) -> bool:
            """Emit tokens until one of the two stopping conditions. False
            means the run is over and the caches no longer matter."""
            for token in tokens:
                if stop_at_eos and eos_id is not None and token == eos_id:
                    return False
                produced.append(token)
                if on_token:
                    on_token(token)
                if len(produced) >= max_tokens:
                    return False
            return True

        # A rejection can push accepted-but-unread tokens back into the
        # next batch (see the module docstring). Left unbounded, a run of
        # rejections would make every batch longer than the last, so past
        # this many the queue is read on its own and cleared.
        max_pending = 2 * (self.k + 1)

        if not keep([first]):
            return produced

        while True:
            if len(pending) > max_pending:
                self.model.forward(pending[:-1], cache, offset=cache.length)
                self.passes += 1
                pending = pending[-1:]

            drafts, distributions = self.drafter.propose(
                list(prompt) + produced, self.k)
            self.proposed += len(drafts)
            self.rounds += 1

            snapshot = self.model.snapshot_state()
            base = cache.length
            batch = pending + drafts
            rows = self.model.forward(batch, cache, offset=base,
                                      all_positions=True)
            self.passes += 1

            head = len(pending) - 1
            n = self._accept(rows, head, drafts, distributions,
                             temperature, top_p, rng)
            self.accepted += n
            bonus = self._bonus(rows[head + n], drafts, distributions, n,
                                temperature, top_p, rng)

            if n == len(drafts):
                # Everything in the batch was wanted, so the pass stands.
                pending = [bonus]
            elif snapshot is None:
                # Attention only: the rejected span is a counter away.
                cache.rollback(base + head + 1 + n)
                pending = [bonus]
            else:
                # Recurrent layers have already absorbed the whole batch,
                # so the pass is undone entirely and its accepted prefix
                # is read again as part of the next one.
                self.model.restore_state(snapshot)
                cache.rollback(base)
                pending = pending + drafts[:n] + [bonus]

            # A round yields the proposals the model agreed with plus the
            # one it produced itself, so k proposals can be k+1 tokens.
            if not keep(drafts[:n] + [bonus]):
                return produced

        return produced

    def _draw(self, logits, temperature, top_p, rng) -> int:
        from reminis.infer import _sample
        return _sample(logits, temperature, top_p, rng)

    def _accept(self, rows, head, drafts, distributions, temperature, top_p, rng):
        """How many leading proposals the target model agrees with.

        Greedy is the simple case and the one worth stating plainly: a
        proposal is kept when it is what argmax would have chosen, so the
        text is identical to a run with no drafter at all.

        Sampling is the rule from the speculative-decoding paper. A
        proposal drawn with probability q is kept with probability
        min(1, p/q) under the target's p, and the token that replaces a
        rejected one is drawn from the difference. Together those give
        exactly the target's distribution -- the draft changes the speed
        and not the model.
        """
        from reminis.infer import _probs

        for j, token in enumerate(drafts):
            if temperature <= 0:
                if int(np.argmax(rows[head + j])) != token:
                    return j
                continue
            p = _probs(rows[head + j], temperature, top_p)[token]
            # No distribution means the proposal was made with certainty,
            # so q is 1 and the acceptance probability is p itself.
            q = 1.0 if distributions is None else float(distributions[j][token])
            if q <= 0.0 or rng.random() >= min(1.0, float(p) / q):
                return j
        return len(drafts)

    def _bonus(self, row, drafts, distributions, n, temperature, top_p, rng):
        """One token from the position where the agreement ran out.

        On a full acceptance this is the ordinary next-token sample, which
        is why k proposals can yield k+1 tokens. On a rejection it is drawn
        from the target's distribution with the draft's removed, which is
        what keeps the whole scheme exact rather than merely close.
        """
        from reminis.infer import _probs

        if temperature <= 0:
            return int(np.argmax(row))
        p = _probs(row, temperature, top_p)
        if n < len(drafts) and distributions is not None:
            residual = p - distributions[n]
            np.maximum(residual, 0.0, out=residual)
            total = residual.sum()
            # The two distributions can agree to within rounding, leaving
            # nothing to draw from. Falling back to the target's own is
            # the right answer there and not an approximation of one.
            if total > 1e-9:
                p = residual / total
        elif n < len(drafts):
            # A point-mass draft: everything but the rejected token.
            p = p.copy()
            p[drafts[n]] = 0.0
            total = p.sum()
            p = p / total if total > 1e-9 else _probs(row, temperature, top_p)
        return int(rng.choice(len(p), p=p))
