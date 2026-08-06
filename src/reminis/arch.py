"""What a model's blocks actually do, per architecture.

Every architecture in this file is a family of small deviations from the
same shape: normalise, attend, normalise, feed forward, add. `infer.py`
implements that shape once. This file holds the deviations, so that adding
a model means describing how it differs rather than editing a forward pass
that seven other models depend on.

An architecture supplies as little or as much as it needs:

  * nothing but a name and a rotary layout, which is the llama family;
  * `prepare_meta`, when the GGUF records a hyperparameter in a form the
    common config parser cannot read -- an array where it expects a number;
  * `configure`, to derive its own fields onto the config;
  * `block`, when the block is genuinely a different computation and
    expressing it as flags on the shared one would be a fiction;
  * `logits`, for what happens after the final norm.

Anything left out falls back to the shared implementation, so a model that
is llama with a different rotary layout costs three lines here.
"""

from __future__ import annotations

import json
import math

import numpy as np

_REGISTRY: dict[str, "Arch"] = {}


def register(cls):
    """Add an architecture, keyed by the name GGUF records for it."""
    _REGISTRY[cls.name] = cls()
    return cls


def get(name: str):
    return _REGISTRY.get(name)


def names() -> list[str]:
    return sorted(_REGISTRY)


class Arch:
    """One architecture's departures from the common block."""

    #: The value of `general.architecture` in the GGUF this implements.
    name = ""

    #: How rotary embedding pairs up channels. "norm" rotates adjacent
    #: pairs (0,1), (2,3); "neox" rotates i against i + rope_dim/2. Applying
    #: the wrong one produces confident gibberish rather than an error.
    rope_style = "norm"

    def prepare_meta(self, meta: dict) -> dict:
        """Substitutions applied to the metadata before the config reads it.

        For hyperparameters that vary per layer, the GGUF holds an array
        where the shared parser wants a scalar. Returning a representative
        scalar here keeps that parser simple; `configure` then reads the
        real arrays and puts the per-layer values where the block can see
        them.
        """
        return {}

    def configure(self, cfg, meta: dict, store) -> None:
        """Derive this architecture's own fields onto the config."""

    def block(self, model, x, layer: int, cache, offset: int):
        """This architecture's block, or None to use the shared one."""
        return None

    def logits(self, model, logits):
        """The last step, after the output projection."""
        return logits


# The llama family and its near neighbours: one shared block, distinguished
# only by how rotary embedding is laid out.
for _name, _style in (
    ("gpt-oss", "neox"),
    ("llama", "norm"),
    ("mistral", "norm"),
    ("granite", "norm"),
    ("granitemoe", "norm"),
    ("qwen2", "neox"),
    ("qwen2moe", "neox"),
):
    register(type(f"_{_name}", (Arch,), {"name": _name, "rope_style": _style}))


def _as_list(value):
    """A GGUF array as a flat Python list.

    These arrive as JSON text holding one-element lists per layer --
    `[[8], [8], [2]]` -- or already parsed, depending on how the value
    made it through conversion.
    """
    if isinstance(value, str):
        value = json.loads(value.replace("True", "true").replace("False", "false"))
    if not isinstance(value, (list, tuple)):
        return [value]
    out = []
    for item in value:
        while isinstance(item, (list, tuple)):
            item = item[0]
        out.append(item)
    return out


@register
class Gemma4(Arch):
    """Gemma 4: alternating attention geometries, and a mixture beside the
    feed-forward rather than in place of it.

    Three things here have no counterpart in the other architectures:

    **The layers are not alike.** One layer in six attends globally with 2
    key/value heads of width 512 and a rotary base of 1e6; the rest see a
    1024-token window with 8 heads of width 256 and a base of 1e4. So head
    count, head width, rotary base and rotary width are all per-layer, and
    the GGUF records them as arrays.

    **The global layers have no value projection.** There is no `attn_v`
    tensor on those layers because value *is* key -- the pre-norm output of
    the key projection, normalised without a learned scale and left
    unrotated. Reading that as a missing weight rather than as a deliberate
    tying gives a model that runs and says nothing.

    **The mixture is additive, not alternative.** Every layer runs a dense
    feed-forward *and* a 128-expert mixture, on separately normalised copies
    of the same input, and adds the two. The mixture's input is the block's
    residual rather than the feed-forward's output, so the two branches are
    parallel rather than sequential.

    Checked against transformers' own `modeling_gemma4.py` driven by these
    weights, layer by layer for all thirty: every layer agrees to a
    correlation of 0.99997 or better, the worst being the global layers,
    where the gap is the four-bit expert index rather than the arithmetic.

    One thing to know before reading its output: this is a reasoning model,
    and on a bare completion prompt it answers by opening a thinking
    channel -- `thought`, `<|channel>` -- which reads exactly like a broken
    forward pass. It is not. Use `--chat` and it answers the question.
    """

    name = "gemma4"
    rope_style = "neox"

    # Layers whose attention spans the whole context; the rest are windowed.
    _GLOBAL = False

    def prepare_meta(self, meta: dict) -> dict:
        """Scalars standing in for the per-layer arrays.

        The shared parser wants one number for head count and head width.
        The widest layer is the right stand-in: it is what the KV cache and
        the fused-weight paths would have to accommodate, and every
        per-layer value is read again in `configure`.
        """
        out = {}
        kv = _as_list(meta.get("gemma4.attention.head_count_kv"))
        if kv:
            out["gemma4.attention.head_count_kv"] = max(int(v) for v in kv)
        # The shared config reads a sliding-window *pattern* as "one layer
        # in every N is global". This model records the pattern explicitly
        # as a bool per layer, so the shared rule is switched off and the
        # array is consulted directly.
        out["gemma4.attention.sliding_window_pattern"] = 0
        return out

    def configure(self, cfg, meta: dict, store) -> None:
        n = cfg.n_layers

        def scalar(key, default):
            value = meta.get(f"gemma4.{key}")
            return default if value is None else type(default)(value)

        sliding = _as_list(meta.get("gemma4.attention.sliding_window_pattern"))
        sliding = [bool(v) for v in sliding] or [False] * n
        sliding = (sliding * n)[:n] if len(sliding) < n else sliding[:n]

        kv_heads = [int(v) for v in _as_list(meta.get("gemma4.attention.head_count_kv"))]
        kv_heads = (kv_heads * n)[:n] if len(kv_heads) < n else kv_heads[:n]

        head_dim_global = scalar("attention.key_length", 256)
        head_dim_swa = scalar("attention.key_length_swa", head_dim_global)
        rope_dim_global = scalar("rope.dimension_count", head_dim_global)
        rope_dim_swa = scalar("rope.dimension_count_swa", rope_dim_global)
        base_global = scalar("rope.freq_base", 1000000.0)
        base_swa = scalar("rope.freq_base_swa", base_global)

        # Everything the block needs to know about a layer, resolved once
        # here rather than re-derived per token.
        cfg.layers = [
            {
                "sliding": sliding[i],
                "kv_heads": kv_heads[i],
                "head_dim": head_dim_swa if sliding[i] else head_dim_global,
                "rope_dim": rope_dim_swa if sliding[i] else rope_dim_global,
                "rope_base": base_swa if sliding[i] else base_global,
                "window": cfg.sliding_window if sliding[i] else 0,
            }
            for i in range(n)
        ]

        # Attention here is unscaled: the query norm carries what 1/sqrt(d)
        # carries elsewhere, so dividing again would halve every score.
        cfg.attn_scale = 1.0

        # Gemma scales the embedding by sqrt(d_model) at run time; the
        # weights are stored unscaled.
        cfg.embedding_scale = math.sqrt(cfg.d_model)

        cfg.logit_softcap = scalar("final_logit_softcapping", 0.0)
        cfg.moe_intermediate = scalar("expert_feed_forward_length", 0)

        # The shared MoE path looks for separate gate and up tensors. This
        # model fuses them, and runs the mixture alongside the dense
        # feed-forward rather than instead of it, so the shared path is
        # switched off and `block` below does both.
        cfg.is_moe = False

    # -- pieces ----------------------------------------------------------

    def _ones(self, model, width: int):
        """A unit norm weight, for the normalisations that have none.

        Two of this model's norms -- the one on values and the one in the
        router -- normalise without a learned scale. Passing ones through
        the backend's kernel keeps them on the same code path as every
        other norm instead of open-coding the arithmetic per backend.
        """
        cache = getattr(model, "_gemma_ones", None)
        if cache is None:
            cache = model._gemma_ones = {}
        arr = cache.get(width)
        if arr is None:
            arr = cache[width] = model.backend.from_numpy(
                np.ones(width, dtype=np.float32)
            )
        return arr

    def _norm(self, model, x, name: str):
        w = model.store.get(name).reshape(-1)
        return model.backend.rms_norm(x, w, model.cfg.rms_eps)

    def _gelu(self, model, x):
        """The tanh approximation of GELU, which is what Gemma trained on."""
        b = model.backend
        xp = b.xp
        inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + xp.tanh(inner))

    def _attention(self, model, x, layer, cache, offset, spec):
        b = model.backend
        cfg = model.cfg
        xp = b.xp
        p = f"blk.{layer}."
        n_tokens = x.shape[0]
        head_dim = spec["head_dim"]

        q = model._linear(x, p + "attn_q.weight")
        k = model._linear(x, p + "attn_k.weight")

        q = q.reshape(n_tokens, cfg.n_heads, head_dim)
        k = k.reshape(n_tokens, spec["kv_heads"], head_dim)

        # Value is the key projection before it is normalised or rotated,
        # on the layers that have no value projection of their own.
        if model.store.has(p + "attn_v.weight"):
            v = model._linear(x, p + "attn_v.weight")
        else:
            v = k
        v = v.reshape(n_tokens, spec["kv_heads"], head_dim)

        q = model.backend.rms_norm(
            q, model.store.get(p + "attn_q_norm.weight").reshape(-1), cfg.rms_eps)
        k_normed = model.backend.rms_norm(
            k, model.store.get(p + "attn_k_norm.weight").reshape(-1), cfg.rms_eps)
        # Values are normalised too, but without a scale to learn.
        v = model.backend.rms_norm(v, self._ones(model, head_dim), cfg.rms_eps)

        q = q.transpose(1, 0, 2)[None]
        k_normed = k_normed.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]

        traditional = self.rope_style == "norm"
        q = b.rope(q, spec["rope_dim"], traditional, spec["rope_base"], offset, None)
        k_normed = b.rope(k_normed, spec["rope_dim"], traditional,
                          spec["rope_base"], offset, None)

        k_all, v_all = cache.append(layer, k_normed, v)

        window = spec["window"]
        mask = None
        if n_tokens > 1 or window:
            mask = model._causal_mask(n_tokens, offset, k_all.shape[-2], window)

        out = b.attention(q, k_all, v_all, cfg.attn_scale, mask, None)
        out = out[0].transpose(1, 0, 2).reshape(n_tokens, cfg.n_heads * head_dim)
        return model._linear(out, p + "attn_output.weight")

    def _route(self, model, h, p: str):
        """Which experts each token uses, and how much of each.

        The router normalises without a scale, applies a learned per-channel
        scale and a fixed 1/sqrt(d_model), and softmaxes in float32 -- the
        reduction that decides which experts run at all, where a rounding
        changes the computation rather than nudging it.
        """
        b = model.backend
        xp = b.xp
        cfg = model.cfg
        k = cfg.n_experts_used

        x = b.rms_norm(h, self._ones(model, cfg.d_model), cfg.rms_eps)
        x = x * model.store.get(p + "ffn_gate_inp.scale").reshape(-1)
        x = x * (cfg.d_model ** -0.5)

        logits = model.backend.matmul_weight(x, model.store.get(p + "ffn_gate_inp.weight"))
        probs = b.softmax(b.to_compute32(logits))

        idx = xp.argpartition(-probs, k - 1, axis=-1)[:, :k]
        weight = xp.take_along_axis(probs, idx, axis=-1)
        weight = weight / xp.sum(weight, axis=-1, keepdims=True)

        # One learned scale per expert, applied to whichever tokens chose it.
        per_expert = model.store.get(p + "ffn_down_exps.scale").reshape(-1)
        weight = weight * xp.take(per_expert, idx.reshape(-1)).reshape(weight.shape)
        return idx, weight

    def _experts(self, model, h, p: str, idx, weight):
        """The chosen experts, fetched one at a time.

        Gemma fuses each expert's gate and up projections into a single
        matrix, so one read yields both halves and they are split after the
        multiply rather than gathered separately.
        """
        b = model.backend
        xp = b.xp
        n_tokens = h.shape[0]
        k = model.cfg.n_experts_used
        chosen = np.asarray(b.to_numpy(idx)).astype(int).reshape(n_tokens, k)
        w = b.to_numpy(weight).reshape(n_tokens, k)

        out = None
        for token in range(n_tokens):
            row = h[token:token + 1]
            acc = None
            for slot in range(k):
                e = int(chosen[token, slot])
                gate_up = model.store.expert(p + "ffn_gate_up_exps.weight", e)
                fused = b.matmul_weight(row, gate_up)
                half = fused.shape[-1] // 2
                gate, up = fused[..., :half], fused[..., half:]
                hidden = self._gelu(model, gate) * up
                down = model.store.expert(p + "ffn_down_exps.weight", e)
                piece = b.matmul_weight(hidden, down) * float(w[token, slot])
                acc = piece if acc is None else acc + piece
            out = acc if out is None else xp.concatenate([out, acc], axis=0)
        return out

    # -- the block -------------------------------------------------------

    def block(self, model, x, layer: int, cache, offset: int):
        cfg = model.cfg
        p = f"blk.{layer}."
        spec = cfg.layers[layer]

        # Attention, normalised on the way in and again on the way out.
        h = self._norm(model, x, p + "attn_norm.weight")
        h = self._attention(model, h, layer, cache, offset, spec)
        h = self._norm(model, h, p + "post_attention_norm.weight")
        x = x + h

        residual = x

        # The dense branch.
        h = self._norm(model, x, p + "ffn_norm.weight")
        gate = model._linear(h, p + "ffn_gate.weight")
        up = model._linear(h, p + "ffn_up.weight")
        dense = model._linear(self._gelu(model, gate) * up, p + "ffn_down.weight")
        dense = self._norm(model, dense, p + "post_ffw_norm_1.weight")

        # The mixture branch, which reads the residual rather than the
        # dense branch's output -- the two are parallel.
        idx, weight = self._route(model, residual, p)
        h2 = self._norm(model, residual, p + "pre_ffw_norm_2.weight")
        h2 = self._experts(model, h2, p, idx, weight)
        h2 = self._norm(model, h2, p + "post_ffw_norm_2.weight")

        h = self._norm(model, dense + h2, p + "post_ffw_norm.weight")
        x = residual + h

        scale = model.store.get(p + "layer_output_scale.weight").reshape(-1)
        return x * scale

    def logits(self, model, logits):
        """Softcap: squash the range with a tanh instead of clipping it."""
        cap = getattr(model.cfg, "logit_softcap", 0.0)
        if not cap:
            return logits
        xp = model.backend.xp
        return xp.tanh(logits / cap) * cap
