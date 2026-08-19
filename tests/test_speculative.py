"""Speculative decoding has to be a speed change and nothing else.

That is the whole claim, and it is testable without a second machine or a
stopwatch: greedy decoding with a drafter must produce the *same token
ids* as greedy decoding without one. Any error in the accept/reject rule,
the cache rollback, or the recurrent-state rollback shows up as a
divergence, usually a few tokens in rather than at the start, which is why
these compare whole sequences instead of the first token.

The cases are chosen for what each one can break:

  * ``ngram`` on a model that is pure attention -- the rollback path where
    a rejected span is a counter away
  * ``ngram`` on qwen35 -- the path where three quarters of the layers are
    recurrent, so the pass cannot be truncated and the accepted prefix has
    to be carried into the next batch instead
  * a real draft *model*, which is the case where the proposals are good
    enough that most rounds accept everything and the boundaries are
    rarely exercised -- so it is run with a deliberately bad draft too
  * a prompt that repeats itself, so the n-gram drafter actually proposes
    something and the acceptance count is not vacuously zero
  * a mismatched tokenizer, which must be refused rather than run slowly
"""

import numpy as np
import pytest

from reminis.backend import select as select_backend
from reminis.errors import UnsupportedModel
from reminis.infer import KVCache, Model, generate
from reminis.speculative import (
    ModelDrafter,
    NgramDrafter,
    Speculator,
    check_compatible,
    open_drafter,
)

from tests.test_qwen35 import TOKENS, build_db

SMOL = "models/SmolLM-135M-Instruct.f16.db"


def _greedy(db, prompt, n, **kw):
    """Token ids from a greedy run, with or without a drafter."""
    return generate(str(db), prompt, max_tokens=n, temperature=0.0,
                    verbose=False, on_token=lambda _p: None,
                    **kw)["token_ids"]


# --------------------------------------------------------------- the drafter


def test_ngram_proposes_the_earlier_continuation():
    """The lookup itself, with no model involved."""
    drafter = NgramDrafter(max_ngram=3, min_ngram=2)
    # "a b c d e" earlier, and the suffix "b c" now: "d e" should follow.
    sequence = [1, 2, 3, 4, 5, 9, 9, 1, 2, 3]
    tokens, dists = drafter.propose(sequence, 2)
    assert tokens == [4, 5]
    assert dists is None


def test_ngram_declines_when_nothing_matches():
    """No match means no proposal, which the loop must handle as a plain step."""
    drafter = NgramDrafter(max_ngram=4, min_ngram=2)
    assert drafter.propose([1, 2, 3, 4, 5, 6], 3) == ([], None)


def test_ngram_never_proposes_from_the_suffix_itself():
    """The last n tokens always match themselves, and following that match
    would propose the tokens that were just emitted, forever."""
    drafter = NgramDrafter(max_ngram=3, min_ngram=2)
    tokens, _ = drafter.propose([7, 8, 9], 3)
    assert tokens == []


# ------------------------------------------------------------- the cache


def test_rollback_restores_the_earlier_span():
    """Rolling back and re-reading must land where reading once does."""
    backend = select_backend("inference", "numpy")
    model = Model(SMOL, backend=backend)
    ids = [1, 2, 3, 4, 5, 6]
    try:
        straight = KVCache(model.cfg.n_layers, capacity=16, backend=backend)
        want = model.forward(ids, straight, offset=0)

        rolled = KVCache(model.cfg.n_layers, capacity=16, backend=backend)
        model.forward(ids[:3], rolled, offset=0)
        # A speculation that is then rejected.
        model.forward([99, 98, 97], rolled, offset=rolled.length)
        assert rolled.length == 6
        rolled.rollback(3)
        got = model.forward(ids[3:], rolled, offset=rolled.length)

        assert np.allclose(want, got, atol=1e-4)
    finally:
        model.close()


def test_rollback_refuses_to_invent_tokens():
    cache = KVCache(2, capacity=8, backend=select_backend("inference", "numpy"))
    with pytest.raises(ValueError):
        cache.rollback(1)


# ------------------------------------------------- same text, fewer passes


REPEATED = (
    "The cat sat on the mat. The dog sat on the log. The cat sat on the mat. "
    "The cat sat on the"
)


def test_ngram_draft_matches_plain_greedy():
    """Attention-only model: the counter-rollback path."""
    plain = _greedy(SMOL, REPEATED, 24)
    drafted = _greedy(SMOL, REPEATED, 24, draft="ngram", draft_tokens=4)
    assert drafted == plain


def test_ngram_draft_actually_accepts_something():
    """A test that passed because nothing was ever proposed would prove
    nothing, so the acceptance count is checked as well as the text."""
    result = generate(SMOL, REPEATED, max_tokens=24, temperature=0.0,
                      verbose=False, on_token=lambda _p: None,
                      draft="ngram", draft_tokens=4)
    assert result["draft_proposed"] > 0
    assert result["draft_accepted"] > 0
    # The point of the exercise: fewer reads of the weights than tokens.
    assert result["model_passes"] < result["generated_tokens"]


def test_model_draft_matches_plain_greedy():
    """A model drafting for itself. Every proposal is right, so this is the
    all-accepted path -- the one where the cache must *not* be rolled back."""
    plain = _greedy(SMOL, "The capital of France is", 16)
    drafted = _greedy(SMOL, "The capital of France is", 16,
                      draft=SMOL, draft_tokens=3)
    assert drafted == plain


def test_model_draft_of_itself_accepts_everything():
    result = generate(SMOL, "The capital of France is", max_tokens=16,
                      temperature=0.0, verbose=False,
                      on_token=lambda _p: None, draft=SMOL, draft_tokens=3)
    assert result["draft_accepted"] == result["draft_proposed"] > 0


def test_sampling_accepts_an_identical_draft_every_time():
    """The accept/reject rule at temperature, where it is not argmax.

    A proposal drawn with probability q is kept with probability min(1, p/q).
    A model drafting for itself has p == q at every position, so the ratio
    is exactly one and nothing may ever be rejected -- which pins the rule
    without needing to sample enough runs to measure a distribution.
    """
    result = generate(SMOL, "The capital of France is", max_tokens=16,
                      temperature=0.9, top_p=0.95, seed=3, verbose=False,
                      on_token=lambda _p: None, draft=SMOL, draft_tokens=3)
    assert result["draft_proposed"] > 0
    assert result["draft_accepted"] == result["draft_proposed"]


def test_a_bad_draft_still_produces_the_right_text():
    """The rejection path, forced. A drafter that proposes nonsense must
    cost speed and nothing else -- if a rejected token could survive, this
    is where it would show."""
    backend = select_backend("inference", "numpy")
    model = Model(SMOL, backend=backend)
    try:
        class Wrong:
            description = "deliberately wrong"

            def propose(self, sequence, k):
                return [(sequence[-1] + 7 + i) % 1000 for i in range(k)], None

            def close(self):
                pass

        ids = model.tokenizer.encode("The capital of France is")
        cache = KVCache(model.cfg.n_layers, capacity=len(ids) + 32,
                        backend=backend)
        logits = model.forward(ids, cache, offset=0)
        spec = Speculator(model, Wrong(), k=3)
        got = spec.generate(logits, ids, cache, 12, 0.0, 1.0,
                            np.random.default_rng(0), eos_id=None)
        assert spec.accepted == 0
    finally:
        model.close()

    assert got == _greedy(SMOL, "The capital of France is", 12)


# ------------------------------------------------------- the recurrent case


def test_recurrent_model_draft_matches_plain_greedy(tmp):
    """qwen35 is three-quarters recurrent, so a rejected pass cannot be
    truncated out of the cache -- the accepted prefix is carried into the
    next batch instead. This is the test for that path.

    The weights are random, so the text is gibberish. It has to be the
    *same* gibberish.
    """
    db = build_db(tmp, name="qwen35-spec.db")
    model = Model(str(db), backend=select_backend("inference", "numpy"))
    try:
        assert model.snapshot_state() is not None, \
            "this architecture is supposed to carry state; the test is moot"
    finally:
        model.close()

    prompt_ids = list(TOKENS)

    def run(**kw):
        model = Model(str(db), backend=select_backend("inference", "numpy"))
        try:
            cache = KVCache(model.cfg.n_layers, capacity=len(prompt_ids) + 32,
                            backend=model.backend)
            logits = model.forward(prompt_ids, cache, offset=0)
            drafter = kw.pop("drafter", None)
            if drafter is None:
                out, rng = [], np.random.default_rng(0)
                for _ in range(12):
                    token = int(np.argmax(logits))
                    out.append(token)
                    logits = model.forward([token], cache, offset=cache.length)
                return out
            spec = Speculator(model, drafter, k=3)
            return spec.generate(logits, prompt_ids, cache, 12, 0.0, 1.0,
                                 np.random.default_rng(0), eos_id=None)
        finally:
            model.close()

    plain = run()

    class Alternating:
        """Right half the time, so both the accept and the reject path run."""
        description = "alternating"

        def __init__(self, truth):
            self.truth, self.calls = truth, 0

        def propose(self, sequence, k):
            self.calls += 1
            base = len(sequence) - len(prompt_ids)
            out = []
            for i in range(k):
                j = base + i
                right = j < len(self.truth) and (self.calls + i) % 2 == 0
                out.append(self.truth[j] if right else 3)
            return out, None

        def close(self):
            pass

    assert run(drafter=Alternating(plain)) == plain


def test_recurrent_state_snapshot_is_not_a_view(tmp):
    """The snapshot has to survive the pass that follows it. If the state
    were written into rather than replaced, restoring would restore
    nothing and the test above would pass for the wrong reason."""
    db = build_db(tmp, name="qwen35-snap.db")
    model = Model(str(db), backend=select_backend("inference", "numpy"))
    try:
        cache = KVCache(model.cfg.n_layers, capacity=32, backend=model.backend)
        model.forward(list(TOKENS), cache, offset=0)
        snapshot = model.snapshot_state()
        before = [None if s is None else np.array(s)
                  for s in model._ssm_states._ssm]

        model.forward([5, 6, 7], cache, offset=cache.length)
        model.restore_state(snapshot)
        cache.rollback(len(TOKENS))

        for was, now in zip(before, model._ssm_states._ssm):
            if was is None:
                assert now is None
            else:
                assert np.array_equal(was, np.asarray(now))
    finally:
        model.close()


# ------------------------------------------------------------ compatibility


def test_mismatched_tokenizer_is_refused(tmp):
    """A draft whose ids mean something else does not fail, it just wastes
    time -- so it has to be caught rather than left to run."""
    db = build_db(tmp, name="qwen35-other.db")
    target = Model(SMOL, backend=select_backend("inference", "numpy"))
    draft = Model(str(db), backend=select_backend("inference", "numpy"))
    try:
        with pytest.raises(UnsupportedModel, match="tokenizer"):
            check_compatible(target, draft)
    finally:
        draft.close()
        target.close()


def test_a_missing_bos_row_is_not_an_incompatibility():
    """Two builds from one family often disagree about whether to record a
    beginning-of-text id. It changes the meaning of no token, and refusing
    over it would reject the one real draft pair on this machine."""
    a = Model(SMOL, backend=select_backend("inference", "numpy"))
    b = Model(SMOL, backend=select_backend("inference", "numpy"))
    try:
        b.meta.pop("tokenizer.ggml.bos_token_id", None)
        check_compatible(a, b)
    finally:
        a.close()
        b.close()


def test_a_different_vocabulary_is_an_incompatibility():
    """One token changed is enough: every id after it means something else."""
    a = Model(SMOL, backend=select_backend("inference", "numpy"))
    b = Model(SMOL, backend=select_backend("inference", "numpy"))
    try:
        b.meta["tokenizer.ggml.tokens"] = b.meta["tokenizer.ggml.tokens"][:-1]
        with pytest.raises(UnsupportedModel, match="tokenizer"):
            check_compatible(a, b)
    finally:
        a.close()
        b.close()


def test_a_model_is_compatible_with_itself():
    a = Model(SMOL, backend=select_backend("inference", "numpy"))
    b = Model(SMOL, backend=select_backend("inference", "numpy"))
    try:
        check_compatible(a, b)
    finally:
        a.close()
        b.close()


def test_open_drafter_reads_the_ngram_spelling():
    assert isinstance(open_drafter("ngram", None, 8, 0.0, 1.0, None),
                      NgramDrafter)


def test_open_drafter_builds_a_model_drafter():
    target = Model(SMOL, backend=select_backend("inference", "numpy"))
    try:
        drafter = open_drafter(SMOL, target, 32, 0.0, 1.0,
                               np.random.default_rng(0),
                               backend=select_backend("inference", "numpy"))
        assert isinstance(drafter, ModelDrafter)
        drafter.close()
    finally:
        target.close()
