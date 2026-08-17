"""The hyperparameters, read from rows rather than from a config file.

A GGUF file names its metadata by architecture -- ``llama.block_count``,
``qwen2.attention.head_count`` -- and a safetensors conversion names it
after the keys in ``config.json``. Both end up as rows in ``model_meta``,
and this is where they become the handful of numbers a forward pass
actually needs: how many layers, how wide, how many heads, how the rotary
tables are built, where the context ends.

Anything the architecture does differently is asked of its ``arch.py``
entry through ``spec.configure``, so this file stays a reader of numbers
rather than a list of special cases.
"""

import math

import numpy as np

from reminis import arch as arch_registry
from reminis.errors import UnsupportedModel
from reminis.weights import WeightStore

# Which architectures this can run, and what each does differently, lives
# in `arch.py`. Adding a model means adding an entry there rather than
# threading another special case through this file.
#
# The rotary layout that comes back with it is not cosmetic:
#   "norm" rotates adjacent pairs (0,1), (2,3), ...   -- llama, mistral
#   "neox" rotates halves,  i with i + head_dim/2     -- qwen2
SUPPORTED_ARCHS = {name: arch_registry.get(name).rope_style
                   for name in arch_registry.names()}


def _yarn_periods(dim, base, scale, orig_context, beta_fast, beta_slow):
    """Rotary periods under YaRN, which stretches long contexts unevenly.

    Simple position interpolation divides every frequency by the same
    factor, which preserves long-range structure and destroys the
    high-frequency detail nearby tokens depend on. YaRN interpolates only
    the slow dimensions, leaves the fast ones alone, and ramps between the
    two -- so `beta_fast` and `beta_slow` name the wavelengths where that
    ramp starts and ends.

    Returned as periods rather than frequencies, since that is what the
    rotary kernels take.
    """
    half = dim // 2
    periods = base ** (np.arange(half, dtype=np.float64) * 2.0 / dim)
    if not orig_context:
        return (periods * scale).astype(np.float32)

    def correction(rotations):
        return (dim * math.log(orig_context / (rotations * 2 * math.pi))
                / (2 * math.log(base)))

    low = math.floor(correction(beta_fast))
    high = math.ceil(correction(beta_slow))
    low, high = max(low, 0), min(high, half - 1)
    if high <= low:
        high = low + 0.001

    # 1 where a dimension is left alone, 0 where it is fully interpolated.
    keep = 1.0 - np.clip(
        (np.arange(half, dtype=np.float64) - low) / (high - low), 0, 1
    )
    inv_freq = (1.0 / (periods * scale)) * (1 - keep) + (1.0 / periods) * keep
    return (1.0 / inv_freq).astype(np.float32)


class Config:
    """The hyperparameters, read out of `model_meta`."""

    def __init__(self, meta: dict, store: WeightStore):
        self.arch = meta.get("general.architecture", "")
        if self.arch not in SUPPORTED_ARCHS:
            from reminis.whisper import is_whisper

            # A speech model is not an unrunnable model, it is a differently
            # runnable one, so say which command runs it rather than sending
            # the reader to the viewer.
            if is_whisper(meta):
                raise UnsupportedModel(
                    "This is a Whisper model, which is an encoder-decoder over "
                    "audio rather than a text model.\n"
                    "Transcribe it instead:  reminis transcribe <db> <audio.wav>"
                )
            raise UnsupportedModel(
                f"reminis run implements {', '.join(sorted(SUPPORTED_ARCHS))}; "
                f"this model's architecture is '{self.arch or 'unknown'}'.\n"
                f"Inspect it instead:  reminis view <db>"
            )
        self.spec = arch_registry.get(self.arch)
        self.rope_style = self.spec.rope_style
        a = self.arch

        # An architecture may record a hyperparameter in a form the parser
        # below cannot read -- an array where it expects a number. It says
        # so here, before anything reads the metadata. The substitutions are
        # for that parser only: `configure` is handed the metadata as it was
        # actually written, since it is the thing that reads the arrays.
        raw_meta = meta
        overrides = self.spec.prepare_meta(meta)
        if overrides:
            meta = {**meta, **overrides}

        def num(key, default=None):
            value = meta.get(f"{a}.{key}")
            if value is None:
                if default is None:
                    raise UnsupportedModel(f"Model metadata is missing {a}.{key}")
                return default
            return float(value) if "." in str(value) or "e" in str(value).lower() else int(value)

        self.n_layers = int(num("block_count"))
        self.d_model = int(num("embedding_length"))
        self.n_heads = int(num("attention.head_count"))
        self.n_kv_heads = int(num("attention.head_count_kv", self.n_heads))
        self.head_dim = int(num("attention.key_length", self.d_model // self.n_heads))
        self.rope_base = float(num("rope.freq_base", 10000.0))
        self.rms_eps = float(num("attention.layer_norm_rms_epsilon", 1e-5))
        self.context_length = int(num("context_length", 2048))
        self.rope_dim = int(num("rope.dimension_count", self.head_dim))

        if self.n_heads % self.n_kv_heads != 0:
            raise UnsupportedModel(
                f"{self.n_heads} attention heads do not divide evenly into "
                f"{self.n_kv_heads} key/value heads"
            )

        # Llama 3 stores per-dimension rotary scaling as a tensor rather than
        # as metadata. Ignoring it silently would break long-context models
        # in a way that only shows up as slightly-wrong text.
        self.rope_factors = (
            store.get_numpy("rope_freqs.weight").ravel()
            if store.has("rope_freqs.weight") else None
        )
        # Llama 3 divides each rotary frequency by a stored factor. Folding
        # that in once here means the kernels take plain frequencies and
        # neither backend needs to know the model does anything unusual.
        self.rope_scaling = str(meta.get(f"{a}.rope.scaling.type", "")).lower()
        self.rope_scale_factor = float(num("rope.scaling.factor", 1.0))
        self.rope_orig_context = int(num("rope.scaling.original_context_length", 0))
        self.yarn_beta_fast = float(num("rope.scaling.yarn_beta_fast", 32.0))
        self.yarn_beta_slow = float(num("rope.scaling.yarn_beta_slow", 1.0))

        half = self.rope_dim // 2
        if self.rope_scaling == "yarn" and self.rope_scale_factor > 1:
            self.rope_freqs = _yarn_periods(
                self.rope_dim, self.rope_base, self.rope_scale_factor,
                self.rope_orig_context, self.yarn_beta_fast, self.yarn_beta_slow,
            )
        elif self.rope_factors is not None:
            base_freqs = 1.0 / (
                self.rope_base ** (np.arange(half, dtype=np.float32) * 2.0 / self.rope_dim)
            )
            self.rope_freqs = 1.0 / (base_freqs / self.rope_factors[:half])
        else:
            self.rope_freqs = None

        # Mixture of experts: a router picks a few of many feed-forward
        # networks per token, and the experts are stored stacked into one
        # 3-D tensor per projection rather than as separate matrices.
        self.n_experts = int(num("expert_count", 0))
        self.n_experts_used = int(num("expert_used_count", 0))
        self.is_moe = self.n_experts > 0 and store.has("blk.0.ffn_gate_exps.weight")

        # Granite scales four things the rest of the llama family leaves at
        # 1, and silently ignoring any of them produces fluent nonsense.
        # attention.scale replaces 1/sqrt(head_dim) outright -- for this
        # model it is 1/64 where the usual formula gives 1/8.
        self.attn_scale = float(num("attention.scale", 0.0)) or (
            1.0 / math.sqrt(self.head_dim)
        )
        # A sliding window limits a layer to the most recent keys. Models
        # that use one usually alternate: some layers see everything, the
        # rest see a window, which is how they keep long contexts affordable.
        self.sliding_window = int(num("attention.sliding_window", 0))
        # Models that alternate rarely record the pattern; two -- one
        # windowed layer for each full one -- is what the metadata omits.
        self.swa_pattern = int(num("attention.sliding_window_pattern", 0)) or (
            2 if self.sliding_window else 0
        )

        # gpt-oss's gated feed-forward is not the usual SwiGLU. It clamps
        # both projections, uses sigmoid(1.702 * gate) rather than the plain
        # sigmoid of SiLU, and multiplies by (up + 1) rather than up. Using
        # the standard form instead produces grammatical text that answers
        # nothing, which is the worst way for this to be wrong.
        self.glu_alpha = 1.702 if self.arch == "gpt-oss" else 1.0
        self.glu_limit = 7.0 if self.arch == "gpt-oss" else 0.0
        self.glu_up_offset = 1.0 if self.arch == "gpt-oss" else 0.0

        self.embedding_scale = float(num("embedding_scale", 1.0))
        self.residual_scale = float(num("residual_scale", 1.0))
        self.logit_scale = float(num("logit_scale", 1.0))

        # Tied embeddings: many small models have no separate output matrix.
        self.tied_output = not store.has("output.weight")

        # Anything this architecture derives for itself, including fields
        # the shared parser above has no notion of.
        self.logit_softcap = 0.0
        self.spec.configure(self, raw_meta, store)
