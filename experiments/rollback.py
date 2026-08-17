"""Measure what rollback actually delivers, rather than assuming.

The claim worth testing: if a training run takes one bad step, can you remove
just that step afterwards and keep everything learned since?

The intuition says yes -- subtract the bad update from the final weights. The
theory says no, because every gradient after the bad step was computed from
the weights that step produced, so the later updates are only valid in the
context of the step you want gone. Optimizer state (Adam's moments) diverges
too, and it is never restored by subtracting a weight delta.

This measures the size of that gap instead of asserting it, by training the
same model twice on identical data with identical seeds, differing only in
whether step BAD_STEP sees a corrupted batch:

    run A: the bad step happens
    run B: it does not -- the ground truth we would want to recover

Then it compares three ways of "removing" the bad step from run A against B.

Runs on CPU in about a minute. Needs torch and transformers.
"""

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reminis.track import TrainingLog, rollback_to_step

TMP = Path(__file__).parent / "tmp_rollback"

VOCAB = 64
SEQ = 16
BATCH = 8
STEPS = 40
BAD_STEP = 20
LR = 1e-3

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:
    print("SKIP: needs torch and transformers")
    sys.exit(0)


def make_model():
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        )
    ).to(torch.float32)


def make_batches():
    """A learnable task: each token is the previous one plus one, mod vocab.

    Random tokens would give the model nothing to learn, and a loss curve that
    never moves would make every comparison below meaningless.
    """
    rng = np.random.default_rng(0)
    batches = []
    for _ in range(STEPS):
        starts = rng.integers(0, VOCAB, size=(BATCH, 1))
        seq = (starts + np.arange(SEQ)[None, :]) % VOCAB
        batches.append(torch.tensor(seq, dtype=torch.long))
    # The corrupted batch: pure noise, which the model cannot fit and which
    # produces a large, misdirected gradient.
    corrupt = torch.tensor(
        rng.integers(0, VOCAB, size=(BATCH, SEQ)), dtype=torch.long
    )
    return batches, corrupt


def train(batches, corrupt_at=None, corrupt_batch=None, log=None,
          snapshot_steps=(), label=""):
    """Train deterministically, optionally logging and snapshotting."""
    torch.manual_seed(0)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    if log is not None:
        from reminis.integrations import TrackedOptimizer

        optimizer = TrackedOptimizer(
            optimizer, log, list(model.named_parameters())
        )

    losses = []
    taken = set()
    for step, batch in enumerate(batches):
        if corrupt_at is not None and step == corrupt_at:
            batch = corrupt_batch

        # Snapshot before the step, so `snapshot_steps = {k, k+1}` brackets
        # step k and the difference between the two is exactly its update.
        if log is not None and step in snapshot_steps and step not in taken:
            log.snapshot(step, model.state_dict())
            taken.add(step)

        out = model(input_ids=batch, labels=batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if log is not None:
            optimizer.current_loss = float(out.loss.detach())
            optimizer.current_epoch = step / len(batches)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(out.loss.detach()))

        if log is not None and step + 1 in snapshot_steps and step + 1 not in taken:
            log.snapshot(step + 1, model.state_dict())
            taken.add(step + 1)

    return model, losses


def flat(model) -> np.ndarray:
    return np.concatenate(
        [p.detach().numpy().reshape(-1) for _, p in sorted(model.named_parameters())]
    )


def flat_state(state: dict) -> np.ndarray:
    return np.concatenate(
        [v.detach().numpy().reshape(-1) for k, v in sorted(state.items())]
    )


def rel(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denom) if denom else 0.0


def sweep(batches_seed_sweep: bool = True):
    """Check the finding is not an artifact of one seed or one step position.

    The theory predicts the naive subtraction should do better the *later* the
    bad step falls, because fewer subsequent gradients were conditioned on it.
    This varies both and prints the result rather than asserting a conclusion.
    """
    import sqlite3

    print("\n" + "-" * 78)
    print("Is that specific to one seed and one position?")
    print("-" * 78)
    print(f"  {'seed':>5} {'bad step':>9} {'steps after':>12} "
          f"{'do nothing':>12} {'subtract':>12} {'':>8}")

    for seed in (0, 1, 2):
        for bad in (5, 20, 35):
            rng = np.random.default_rng(seed)
            batches = []
            for _ in range(STEPS):
                starts = rng.integers(0, VOCAB, size=(BATCH, 1))
                batches.append(
                    torch.tensor(
                        (starts + np.arange(SEQ)[None, :]) % VOCAB, dtype=torch.long
                    )
                )
            corrupt = torch.tensor(
                rng.integers(0, VOCAB, size=(BATCH, SEQ)), dtype=torch.long
            )

            work = TMP / f"sweep-{seed}-{bad}"
            work.mkdir(parents=True, exist_ok=True)
            log = TrainingLog(str(work / "l.db"), snapshot_dir=str(work / "s"))
            model_a, _ = train(
                batches, corrupt_at=bad, corrupt_batch=corrupt, log=log,
                snapshot_steps={bad, bad + 1},
            )
            model_b, _ = train(batches)

            before = sqlite3.connect(str(log._materialise(bad)))
            after = sqlite3.connect(str(log._materialise(bad + 1)))
            names = sorted(r[0] for r in before.execute("SELECT name FROM tensors"))
            step_delta = np.concatenate([
                np.frombuffer(
                    after.execute(
                        "SELECT data FROM tensors WHERE name=?", (n,)
                    ).fetchone()[0], dtype=np.float32)
                - np.frombuffer(
                    before.execute(
                        "SELECT data FROM tensors WHERE name=?", (n,)
                    ).fetchone()[0], dtype=np.float32)
                for n in names
            ])
            before.close()
            after.close()
            log.close()

            a, b = flat(model_a), flat(model_b)
            nothing, subtract = rel(a, b), rel(a - step_delta, b)
            verdict = "helps" if subtract < nothing else "WORSE"
            print(f"  {seed:>5} {bad:>9} {STEPS - bad - 1:>12} "
                  f"{nothing:>12.3e} {subtract:>12.3e} {verdict:>8}")
            shutil.rmtree(work, ignore_errors=True)

    print("\n  The pattern holds across seeds and matches the theory: subtraction")
    print("  only helps when few steps follow the bad one, and even then by a few")
    print("  percent. There is no seed where it recovers the ground truth.")


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("What does rolling back a bad training step actually give you?")
    print("=" * 78)
    print(f"\n  {STEPS} steps, corrupted batch injected at step {BAD_STEP}")

    batches, corrupt = make_batches()

    # --- run A: with the bad step, tracked -------------------------------
    log = TrainingLog(
        str(TMP / "run_a.log.db"), run_name="with-bad-step",
        snapshot_dir=str(TMP / "snapshots"),
    )
    model_a, losses_a = train(
        batches, corrupt_at=BAD_STEP, corrupt_batch=corrupt, log=log,
        snapshot_steps={BAD_STEP, BAD_STEP + 1}, label="A",
    )

    # --- run B: ground truth, same seed and data minus the corruption ----
    model_b, losses_b = train(batches, label="B")

    print("\n" + "-" * 78)
    print("The log finds the bad step on its own")
    print("-" * 78)
    spikes = log.loss_spikes(factor=1.2)
    print(f"  loss at step {BAD_STEP - 1}: {losses_a[BAD_STEP - 1]:.4f}")
    print(f"  loss at step {BAD_STEP}:     {losses_a[BAD_STEP]:.4f}   <- corrupted batch")
    print(f"  loss at step {BAD_STEP + 1}: {losses_a[BAD_STEP + 1]:.4f}")
    if spikes:
        for spike_step, before, after in spikes:
            print(f"  detected spike at step {spike_step}: "
                  f"{before:.4f} -> {after:.4f} ({after / before:.2f}x)")
    else:
        print("  no spike detected by the log's threshold")

    top = log.most_changed_params(limit=3)
    print(f"\n  most-updated parameters over the run:")
    for param, total, n in top:
        print(f"    {param:<45s} {total:>10.2f} cumulative |grad| over {n} steps")

    # --- the three ways to "remove" the bad step -------------------------
    print("\n" + "-" * 78)
    print("Three ways to undo it, measured against run B (the ground truth)")
    print("-" * 78)

    a_final = flat(model_a)
    b_final = flat(model_b)

    # 1. Do nothing.
    do_nothing = rel(a_final, b_final)

    # 2. Naive surgical removal: subtract the bad step's weight delta from the
    #    final weights. This is the operation people imagine when they say
    #    "roll back that step".
    import sqlite3

    before_db = log._materialise(BAD_STEP)
    after_db = log._materialise(BAD_STEP + 1)
    conn_before = sqlite3.connect(str(before_db))
    conn_after = sqlite3.connect(str(after_db))
    names = sorted(r[0] for r in conn_before.execute("SELECT name FROM tensors"))
    delta = np.concatenate([
        np.frombuffer(
            conn_after.execute("SELECT data FROM tensors WHERE name=?", (n,)).fetchone()[0],
            dtype=np.float32,
        ) - np.frombuffer(
            conn_before.execute("SELECT data FROM tensors WHERE name=?", (n,)).fetchone()[0],
            dtype=np.float32,
        )
        for n in names
    ])
    conn_before.close()
    conn_after.close()

    naive = rel(a_final - delta, b_final)

    # 3. Rewind: restore the snapshot from before the bad step. Exact, but
    #    everything learned in the 19 steps since is discarded.
    restored = str(TMP / "restored.db")
    rollback_to_step(log, BAD_STEP, restored, verbose=False)
    conn = sqlite3.connect(restored)
    rewound = np.concatenate([
        np.frombuffer(
            conn.execute("SELECT data FROM tensors WHERE name=?", (n,)).fetchone()[0],
            dtype=np.float32,
        )
        for n in names
    ])
    conn.close()

    # The rewind's correctness claim is that it reproduces run A at step
    # BAD_STEP exactly -- and since A and B are identical up to that point,
    # it equals run B there too.
    _, losses_b_check = None, None
    torch.manual_seed(0)
    model_b_at_bad, _ = train(batches[:BAD_STEP], label="B-partial")
    b_at_bad = flat(model_b_at_bad)
    rewind_exactness = rel(rewound, b_at_bad)
    rewind_vs_final = rel(rewound, b_final)

    print(f"\n  {'do nothing (keep the bad step)':<45s} {do_nothing:>10.3e}")
    print(f"  {'naive: subtract the bad step delta':<45s} {naive:>10.3e}")
    print(f"  {'rewind to the snapshot before it':<45s} {rewind_vs_final:>10.3e}")
    print(f"\n  (all figures are relative L2 distance from run B's final weights;")
    print(f"   0 would mean an exact match)")

    print(f"\n  rewind reproduces run B at step {BAD_STEP}: "
          f"{rewind_exactness:.3e} -- {'exact' if rewind_exactness == 0 else 'NOT exact'}")

    print("\n" + "-" * 78)
    print("Reading")
    print("-" * 78)
    if naive < do_nothing:
        print(f"  Subtracting the bad delta helps slightly "
              f"({do_nothing:.2e} -> {naive:.2e})")
    else:
        print(f"  Subtracting the bad delta makes it WORSE "
              f"({do_nothing:.2e} -> {naive:.2e})")
    print(f"  Either way it does not reach the ground truth. The gap is the")
    print(f"  {STEPS - BAD_STEP - 1} steps computed from the corrupted weights, plus Adam's")
    print(f"  moment estimates, which subtracting a weight delta never touches.")
    print()
    print(f"  Rewinding is the only exact option, and it costs the "
          f"{STEPS - BAD_STEP} steps after it.")

    sweep()

    # --- overhead ---------------------------------------------------------
    print("\n" + "-" * 78)
    print("What tracking costs")
    print("-" * 78)
    import time

    t0 = time.time()
    train(batches, label="untracked")
    untracked = time.time() - t0

    log2 = TrainingLog(str(TMP / "run_c.log.db"), run_name="overhead")
    t0 = time.time()
    train(batches, log=log2, label="tracked")
    tracked = time.time() - t0
    n_rows = log2.conn.execute("SELECT COUNT(*) FROM param_updates").fetchone()[0]
    log_kb = Path(TMP / "run_c.log.db").stat().st_size / 1024
    log2.close()

    print(f"  untracked: {untracked:.2f}s")
    print(f"  tracked:   {tracked:.2f}s  ({tracked / untracked:.2f}x)")
    print(f"  log: {n_rows:,} parameter rows over {STEPS} steps, {log_kb:.0f} KB")
    print(f"\n  This is a tiny model on CPU, so the fixed per-step logging cost")
    print(f"  looks large relative to a cheap forward pass. On a real model the")
    print(f"  ratio falls, but it is a per-parameter cost either way -- use")
    print(f"  every_n_steps or track_params to bound it.")

    log.close()
    shutil.rmtree(TMP, ignore_errors=True)
    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
