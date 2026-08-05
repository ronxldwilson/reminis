"""Verify training-log capture, snapshot chaining, and verified rollback.

Trains a real (small) model for real steps rather than synthesising a log, so
the integration surface -- optimizer wrapper, gradient capture timing, snapshot
delta chain -- is exercised the way a user would hit it.
"""

import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reminis.diff import _weights_hash
from reminis.track import TrainingLog, rollback_to_step, state_dict_to_sqlite

TMP = Path(__file__).parent / "tmp_track"

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:
    print("SKIP: needs torch and transformers")
    sys.exit(0)

from reminis.integrations import TrackedOptimizer

STEPS = 12
SNAPSHOT_EVERY = 4


def build():
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        )
    ).to(torch.float32)


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Training log, snapshots, and rollback")
    print("=" * 78)

    model = build()
    log = TrainingLog(
        str(TMP / "run.log.db"), run_name="test-run", snapshot_dir=str(TMP / "snaps")
    )
    optimizer = TrackedOptimizer(
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        log,
        list(model.named_parameters()),
    )

    rng = np.random.default_rng(0)
    weights_at_step = {}

    for step in range(STEPS):
        if step % SNAPSHOT_EVERY == 0:
            log.snapshot(step, model.state_dict())
            weights_at_step[step] = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }

        starts = rng.integers(0, 64, size=(4, 1))
        batch = torch.tensor((starts + np.arange(16)[None, :]) % 64, dtype=torch.long)
        out = model(input_ids=batch, labels=batch)
        out.loss.backward()
        optimizer.current_loss = float(out.loss.detach())
        optimizer.current_epoch = step / STEPS
        optimizer.step()
        optimizer.zero_grad()

    n_params = len(list(model.named_parameters()))

    print("\n" + "-" * 78)
    print("What got logged")
    print("-" * 78)
    n_steps = log.conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
    n_rows = log.conn.execute("SELECT COUNT(*) FROM param_updates").fetchone()[0]
    n_snaps = log.conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    print(f"  steps: {n_steps}, parameter rows: {n_rows}, snapshots: {n_snaps}")

    assert n_steps == STEPS, f"expected {STEPS} steps, logged {n_steps}"
    assert n_rows == STEPS * n_params, \
        f"expected {STEPS * n_params} rows ({n_params} params x {STEPS} steps), got {n_rows}"
    assert n_snaps == len(range(0, STEPS, SNAPSHOT_EVERY))

    # Every row must carry a real gradient norm: capturing gradients after
    # zero_grad() would silently log zeros everywhere, which is the failure
    # this whole integration exists to avoid.
    zero_grads = log.conn.execute(
        "SELECT COUNT(*) FROM param_updates WHERE grad_norm IS NULL OR grad_norm = 0"
    ).fetchone()[0]
    print(f"  rows with a zero/missing gradient: {zero_grads}")
    assert zero_grads == 0, "gradients were captured after they had been zeroed"

    losses = log.loss_curve()
    assert len(losses) == STEPS and all(l is not None for _, l in losses)
    print(f"  loss: {losses[0][1]:.4f} -> {losses[-1][1]:.4f}")
    assert losses[-1][1] < losses[0][1], "model did not learn; fixture is broken"

    top = log.most_changed_params(limit=3)
    print(f"  most-updated: {top[0][0]} ({top[0][1]:.2f} cumulative |grad|)")

    print("\n" + "-" * 78)
    print("Snapshot chain")
    print("-" * 78)
    snaps = log.conn.execute(
        "SELECT step, kind, base_step, bytes FROM snapshots ORDER BY step"
    ).fetchall()
    for step, kind, base_step, size in snaps:
        against = f" against step {base_step}" if base_step is not None else ""
        print(f"  step {step:>3}: {kind}{against}, {size / 1024:.0f} KB")

    assert snaps[0][1] == "full", "the first snapshot cannot be a delta"
    for step, kind, base_step, _ in snaps[1:]:
        assert base_step is None or base_step < step, \
            f"snapshot {step} references a base that is not earlier ({base_step})"

    print("\n" + "-" * 78)
    print("Rollback reproduces the weights exactly")
    print("-" * 78)
    for step in sorted(weights_at_step):
        out_db = str(TMP / f"restored-{step}.db")
        rollback_to_step(log, step, out_db, verbose=False)

        # Compare against an independent database built from the tensors we
        # held in memory at that moment, rather than trusting the log's own
        # bookkeeping to check itself.
        expected_db = str(TMP / f"expected-{step}.db")
        state_dict_to_sqlite(weights_at_step[step], expected_db)

        a = sqlite3.connect(out_db)
        b = sqlite3.connect(expected_db)
        got, want = _weights_hash(a), _weights_hash(b)
        a.close()
        b.close()
        status = "exact" if got == want else "MISMATCH"
        print(f"  step {step:>3}: {status}  ({got[:16]}...)")
        assert got == want, f"rollback to step {step} did not reproduce the weights"

    print("\n" + "-" * 78)
    print("Guards")
    print("-" * 78)
    try:
        rollback_to_step(log, 999, str(TMP / "nope.db"), verbose=False)
    except ValueError as e:
        assert "No snapshot at step 999" in str(e)
        print("  rolling back to a step with no snapshot is refused")
    else:
        raise AssertionError("expected a refusal for a missing snapshot")

    marked = log.mark_rolled_back([4])
    print(f"  marked {marked} parameter rows at step 4 as rolled back")
    assert marked == n_params

    log.close()
    shutil.rmtree(TMP, ignore_errors=True)
    print("\n" + "=" * 78)
    print("ALL TRACKING TESTS PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
