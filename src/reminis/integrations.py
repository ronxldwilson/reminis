"""Hooks that attach a TrainingLog to a real training loop.

Two entry points, because there are two ways people train:

* ``TrackedOptimizer`` wraps any ``torch.optim.Optimizer``. It is the primary
  mechanism, because ``optimizer.step()`` is the one moment where gradients
  still exist *and* the weights have not moved yet -- everything the log wants
  is available in a single place, with no framework cooperation needed.
* ``ReminisCallback`` is a ``TrainerCallback`` for HuggingFace ``Trainer``. It
  cannot see gradients reliably (``Trainer`` zeroes them before most callback
  hooks fire), so it handles loss, epoch, and learning rate, and defers the
  gradient statistics to the wrapped optimizer.

Use them together: pass a ``TrackedOptimizer`` to ``Trainer`` via
``optimizers=`` and add the callback. Either works alone.

torch is imported lazily so importing reminis never requires it.
"""

import time


class TrackedOptimizer:
    """Wrap an optimizer so every step is recorded to a TrainingLog.

    Delegates everything it does not override, so it can be handed to code
    that expects a real optimizer -- including HuggingFace ``Trainer``.

    Args:
        optimizer: The optimizer to wrap.
        log: A ``TrainingLog`` to write to.
        named_parameters: ``model.named_parameters()`` or any iterable of
            (name, parameter). Needed because an optimizer only knows its
            parameters by identity, not by name.
        every_n_steps: Record statistics every N steps. This is the main dial
            for cost. Measured on SmolLM2-135M (134.5M parameters, fp32, CPU):
            +32% per step at 1, +0.5% at 5. The work is a few reductions per
            parameter, so on a GPU -- where real training happens -- it is far
            cheaper than these CPU figures suggest.
        track_params: Optional predicate ``name -> bool`` limiting what is
            recorded. For a LoRA run, restricting this to trainable parameters
            makes the log a small fraction of the size.
        track_weights: Also record weight norms after each step. Roughly half
            the cost, since it doubles the tensors reduced. Turn it off if only
            gradient behaviour matters.
    """

    def __init__(
        self,
        optimizer,
        log,
        named_parameters,
        every_n_steps: int = 1,
        track_params=None,
        track_weights: bool = True,
    ):
        self._optimizer = optimizer
        self._log = log
        self._every_n = max(1, int(every_n_steps))
        self._track_weights = track_weights
        self._named = [
            (name, param)
            for name, param in named_parameters
            if track_params is None or track_params(name)
        ]
        self.step_count = 0
        # Filled in by ReminisCallback when Trainer is driving, so the
        # per-step row can carry loss and LR it would not otherwise see.
        self.current_loss = None
        self.current_epoch = None
        self._overhead = 0.0

    def step(self, *args, **kwargs):
        should_log = self.step_count % self._every_n == 0

        grads = {}
        if should_log:
            t0 = time.time()
            for name, param in self._named:
                if param.grad is not None:
                    grads[name] = param.grad.detach()
            self._overhead += time.time() - t0

        result = self._optimizer.step(*args, **kwargs)

        if should_log and grads:
            t0 = time.time()
            weights = (
                {name: p.detach() for name, p in self._named if name in grads}
                if self._track_weights
                else None
            )
            self._log.log_step(
                step=self.step_count,
                named_grads=grads,
                named_weights=weights,
                loss=self.current_loss,
                epoch=self.current_epoch,
                learning_rate=self._current_lr(),
            )
            self._overhead += time.time() - t0

        self.step_count += 1
        return result

    def _current_lr(self):
        groups = getattr(self._optimizer, "param_groups", None)
        if groups:
            return float(groups[0].get("lr", 0.0))
        return None

    @property
    def logging_overhead_seconds(self) -> float:
        """Wall-clock time spent logging, so the cost can be reported honestly."""
        return self._overhead

    # Everything else is the wrapped optimizer's job.
    def __getattr__(self, name):
        return getattr(self._optimizer, name)

    def __repr__(self):
        return f"TrackedOptimizer({self._optimizer!r})"


def make_callback(log, snapshot_every: int | None = None, tracked_optimizer=None):
    """Build a HuggingFace ``TrainerCallback`` bound to a log.

    Built inside a function because ``TrainerCallback`` is a transformers
    class, and reminis must import cleanly without transformers installed.

    Args:
        log: The ``TrainingLog`` to write to.
        snapshot_every: Take a weight snapshot every N steps. ``None``
            disables snapshots.
        tracked_optimizer: The ``TrackedOptimizer`` in use, if any, so loss and
            epoch reach the per-step rows it writes.
    """
    from transformers import TrainerCallback

    class ReminisCallback(TrainerCallback):
        def __init__(self):
            self.log = log
            self.snapshot_every = snapshot_every
            self.tracked = tracked_optimizer
            self.snapshots = []

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.log.set_meta(
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=args.per_device_train_batch_size,
                learning_rate=args.learning_rate,
                model_class=type(model).__name__ if model is not None else "unknown",
            )
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            # Trainer reports loss here rather than at step end, so this is
            # where the optimizer wrapper learns what to attach to its rows.
            if logs and self.tracked is not None:
                if "loss" in logs:
                    self.tracked.current_loss = float(logs["loss"])
                self.tracked.current_epoch = float(state.epoch or 0.0)
            return control

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if (
                self.snapshot_every
                and model is not None
                and state.global_step > 0
                and state.global_step % self.snapshot_every == 0
            ):
                info = self.log.snapshot(state.global_step, model.state_dict())
                self.snapshots.append(info)
            return control

        def on_train_end(self, args, state, control, model=None, **kwargs):
            self.log.set_meta(
                final_step=state.global_step,
                snapshots_taken=len(self.snapshots),
            )
            if self.tracked is not None:
                self.log.set_meta(
                    logging_overhead_seconds=round(
                        self.tracked.logging_overhead_seconds, 3
                    )
                )
            return control

    return ReminisCallback()
