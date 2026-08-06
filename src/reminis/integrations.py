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


def setup_unsloth(model, log, optimizer=None, snapshot_every=None, every_n_steps=1,
                  track_params=None, track_weights=True):
    """Wire reminis tracking into an unsloth training run.

    Unsloth wraps SFTTrainer, which is a HuggingFace Trainer, so the existing
    TrackedOptimizer and TrainerCallback both work. This function handles the
    wiring and returns the pieces the caller passes to SFTTrainer.

    Args:
        model: The model returned by FastLanguageModel.get_peft_model() or
            similar. Its named_parameters() are used for gradient capture.
        log: A TrainingLog instance.
        optimizer: A torch.optim.Optimizer. If None, the caller must let
            SFTTrainer create its own and pass the returned TrackedOptimizer
            via the ``optimizers=`` argument.
        snapshot_every: Take a snapshot every N steps (None disables).
        every_n_steps: How often to record gradient statistics (default 1).
        track_params: Optional predicate ``name -> bool`` to limit recording.
            For a LoRA run, ``lambda n: 'lora' in n.lower()`` keeps the log
            focused on trainable parameters.
        track_weights: Also record weight norms (default True).

    Returns:
        A dict with keys:
            ``tracked_optimizer``: the TrackedOptimizer wrapping the optimizer,
                or None if no optimizer was provided.
            ``callback``: a TrainerCallback to pass to SFTTrainer's
                ``callbacks=`` list.

    Example::

        from unsloth import FastLanguageModel
        from reminis.track import TrainingLog
        from reminis.integrations import setup_unsloth

        model, tokenizer = FastLanguageModel.from_pretrained(...)
        model = FastLanguageModel.get_peft_model(model, ...)

        log = TrainingLog("run.log.db")
        hooks = setup_unsloth(model, log, snapshot_every=100,
                              track_params=lambda n: "lora" in n.lower())

        trainer = SFTTrainer(
            model=model,
            ...,
            optimizers=(hooks["tracked_optimizer"], None) if hooks["tracked_optimizer"] else (None, None),
            callbacks=[hooks["callback"]],
        )
        trainer.train()
        log.close()
    """
    tracked = None
    if optimizer is not None:
        tracked = TrackedOptimizer(
            optimizer, log, list(model.named_parameters()),
            every_n_steps=every_n_steps,
            track_params=track_params,
            track_weights=track_weights,
        )

    callback = make_callback(
        log, snapshot_every=snapshot_every, tracked_optimizer=tracked,
    )

    return {"tracked_optimizer": tracked, "callback": callback}
