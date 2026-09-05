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
import re

import numpy as np

_REGISTRY: dict[str, "Arch"] = {}


def register(cls):
    """Add an architecture, keyed by the name GGUF records for it."""
    _REGISTRY[cls.name] = cls()
    return cls


def get(name: str):
    return _REGISTRY.get(name)


def from_hf_name(hf_name: str):
    """Look up an arch by its HuggingFace class name or model_type.

    Returns ``(reminis_name, arch)`` or ``(None, None)`` if unrecognised.
    """
    for arch in _REGISTRY.values():
        if hf_name in arch.hf_names:
            return arch.name, arch
    return None, None


def names() -> list[str]:
    return sorted(_REGISTRY)


class Arch:
    """One architecture's departures from the common block."""

    #: The value of `general.architecture` in the GGUF this implements.
    name = ""

    #: HuggingFace architecture class names that map to this reminis arch.
    #: Used by the safetensors converter to resolve ``config.architectures``
    #: and ``config.model_type`` to the internal name.
    hf_names: tuple[str, ...] = ()

    #: Which pre-tokenizer splits text for this architecture. safetensors
    #: carries no equivalent of GGUF's `tokenizer.ggml.pre`, and the wrong
    #: splitter is silent -- the ids stay valid, they are simply not the
    #: ids the model was trained on -- so it is named here.
    pretokenizer: str = ""

    #: How rotary embedding pairs up channels. "norm" rotates adjacent
    #: pairs (0,1), (2,3); "neox" rotates i against i + rope_dim/2. Applying
    #: the wrong one produces confident gibberish rather than an error.
    rope_style = "norm"

    #: The llama-family safetensors block, in the spelling transformers uses,
    #: against the one GGUF uses. Longest fragment wins, so `mlp.gate_proj`
    #: beats `mlp.gate`, and whatever follows the fragment -- `.weight`,
    #: `.bias` -- is carried across untouched.
    _HF_LAYER_MAP = {
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "self_attn.q_norm": "attn_q_norm",
        "self_attn.k_norm": "attn_k_norm",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
        "input_layernorm": "attn_norm",
        "post_attention_layernorm": "ffn_norm",
    }

    #: The tensors that sit outside the repeating block.
    _HF_GLOBAL_MAP = {
        "model.embed_tokens": "token_embd",
        "model.norm": "output_norm",
        "lm_head": "output",
    }

    def translate_name(self, hf_name: str) -> str | None:
        """Map a HuggingFace tensor name to its GGUF equivalent.

        Returns the translated name, or None to keep the original. The
        default handles the llama-family layout -- ``model.layers.N.`` into
        ``blk.N.``, and the projection and norm spellings that go with it.
        Architectures with non-standard tensor names override this.

        A name with no mapping is kept rather than guessed at. It will then
        be missing under the name the model asks for, and `reminis run` says
        which tensor it wanted -- which is a better failure than inventing a
        translation and loading the wrong weight into the right slot.
        """
        for hf_prefix, gguf_name in self._HF_GLOBAL_MAP.items():
            if hf_name == hf_prefix or hf_name.startswith(hf_prefix + "."):
                return gguf_name + hf_name[len(hf_prefix):]

        m = re.match(r"model\.layers\.(\d+)\.(.*)", hf_name)
        if not m:
            return None
        layer, rest = m.group(1), m.group(2)

        for hf_frag, gguf_frag in sorted(
            self._HF_LAYER_MAP.items(), key=lambda kv: -len(kv[0])
        ):
            if rest == hf_frag or rest.startswith(hf_frag + "."):
                return f"blk.{layer}.{gguf_frag}{rest[len(hf_frag):]}"
        return None

    def translate_config(self, config: dict) -> list[tuple[str, str, str]]:
        """Map config.json fields to GGUF-style metadata rows.

        Returns a list of ``(key, value, type)`` tuples. The default covers
        the hyperparameters `ModelConfig` reads for a llama-family block.

        Three of these have defaults in the reader that are wrong for most
        models -- rope's base is 10000 where Qwen2 uses 1000000, and the RMS
        epsilon is 1e-5 where it uses 1e-6 -- and being wrong there is not an
        error, it is slightly wrong text. So they are written out from
        config.json rather than left to fall back.
        """
        tc = config.get("text_config") or config
        a = self.name
        rows = []

        def add(key, value):
            if value is not None:
                rows.append((f"{a}.{key}", str(value), "string"))

        add("block_count", tc.get("num_hidden_layers"))
        add("embedding_length", tc.get("hidden_size"))
        add("feed_forward_length", tc.get("intermediate_size"))
        add("attention.head_count", tc.get("num_attention_heads"))
        add("attention.head_count_kv", tc.get("num_key_value_heads"))
        add("attention.key_length", tc.get("head_dim"))
        add("attention.layer_norm_rms_epsilon", tc.get("rms_norm_eps"))
        add("context_length", tc.get("max_position_embeddings"))

        # rope_theta moved into a nested dict in newer configs; older ones
        # keep it at the top level.
        rope = tc.get("rope_parameters") or {}
        add("rope.freq_base", tc.get("rope_theta", rope.get("rope_theta")))

        scaling = tc.get("rope_scaling") or {}
        if isinstance(scaling, dict) and scaling:
            kind = scaling.get("rope_type") or scaling.get("type")
            add("rope.scaling.type", kind)
            add("rope.scaling.factor", scaling.get("factor"))
            add("rope.scaling.original_context_length",
                scaling.get("original_max_position_embeddings"))

        # Mixture of experts, where the model has one.
        add("expert_count", tc.get("num_experts", tc.get("num_local_experts")))
        add("expert_used_count", tc.get("num_experts_per_tok"))

        return rows

    def needs_transform(self, name: str) -> bool:
        """True when ``transform_tensor`` would change this tensor's data."""
        return False


    def transform_tensor(self, name: str, data: np.ndarray) -> np.ndarray:
        """Apply any data transformations a tensor needs after dequantization.

        Called with the GGUF-convention name, after name translation and
        dequantization. Default is identity.
        """
        return data

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

    def snapshot_state(self, model):
        """Whatever this architecture carries between calls, saved.

        Attention keeps its history in the key/value cache, which can be
        truncated, so a purely attentional architecture has nothing here
        and returns None -- which is also the signal to a caller that
        `KVCache.rollback` is the whole of undoing a forward pass.

        A recurrent architecture is the other case: its history is folded
        into a hidden state with no per-token record left to truncate, so
        the only way back is to have kept a copy.
        """
        return None

    def restore_state(self, model, snapshot) -> None:
        """Put back what `snapshot_state` returned."""


# The llama family and its near neighbours: one shared block, distinguished
# only by how rotary embedding is laid out.
#
# `hf_names` is what lets the safetensors converter recognise a config.json,
# and recognising it is what runs the name and metadata translation above --
# so an architecture left with no names here converts to a database that
# stores fine and cannot be run. That is deliberate for the ones that have
# it. The pre-tokenizer is the reason: an unset one falls back to GPT-2's
# splitter, which produces valid ids that are not the ids the model was
# trained on, and no error anywhere. Naming an architecture here without
# knowing its splitter would trade a loud failure for a silent one, so the
# list holds only what has been run and checked against a reference
# implementation. Adding one is a small job -- name it, name its splitter,
# and extend tests/test_safetensors_run.py -- and is worth doing per model
# someone actually has.
for _name, _style, _hf, _pre in (
    ("gpt-oss", "neox", (), ""),
    ("llama", "norm", (), ""),
    ("mistral", "norm", (), ""),
    ("granite", "norm", (), ""),
    ("granitemoe", "norm", (), ""),
    ("qwen2", "neox", ("Qwen2ForCausalLM", "qwen2"), "qwen2"),
    ("qwen2moe", "neox", (), ""),
):
    register(type(f"_{_name}", (Arch,), {
        "name": _name,
        "rope_style": _style,
        "hf_names": _hf,
        "pretokenizer": _pre,
    }))


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

        Only one thing here is allowed to leave the device: the expert
        numbers, because a blob cannot be read without them. Everything
        else -- the routing weights above all -- stays where it is. Pulling
        each weight across as a Python float cost a device synchronisation
        per expert per layer, which on this model is 240 of them per token,
        and the token has nothing else to wait for while they happen.
        """
        b = model.backend
        xp = b.xp
        n_tokens = h.shape[0]
        k = model.cfg.n_experts_used
        chosen = np.asarray(b.to_numpy(idx)).astype(int).reshape(n_tokens, k)

        # Every read this layer needs is known once the routing is, so ask
        # for all of them before waiting on the first.
        model.store.prefetch_experts(
            (f"{p}{part}.weight", int(e))
            for e in np.unique(chosen)
            for part in ("ffn_gate_up_exps", "ffn_down_exps")
        )

        # Which tokens each expert was chosen by. Decoding routes one token
        # and the grouping is a formality; a prompt routes all of them at
        # once, and then an expert two tokens share is one matrix multiply
        # over two rows rather than two over one row each. At 128 tokens
        # that is the difference between 1,024 multiplies against this
        # layer's experts and at most 128 of them.
        groups = {}
        for token in range(n_tokens):
            for slot in range(k):
                groups.setdefault(int(chosen[token, slot]), []).append((token, slot))

        acc = [None] * n_tokens
        for expert, pairs in groups.items():
            if len(pairs) == 1:
                token, slot = pairs[0]
                rows = h[token:token + 1]
            else:
                rows = xp.take(
                    h, xp.array(np.array([t for t, _ in pairs], dtype=np.int32)),
                    axis=0)

            fused = b.matmul_weight(
                rows, model.store.expert(p + "ffn_gate_up_exps.weight", expert))
            half = fused.shape[-1] // 2
            hidden = self._gelu(model, fused[..., :half]) * fused[..., half:]
            out = b.matmul_weight(
                hidden, model.store.expert(p + "ffn_down_exps.weight", expert))

            for i, (token, slot) in enumerate(pairs):
                # The routing weight as a slice of the device array, not as
                # a number this process has looked at.
                piece = out[i:i + 1] * weight[token, slot]
                acc[token] = piece if acc[token] is None else acc[token] + piece

        return acc[0] if n_tokens == 1 else xp.concatenate(acc, axis=0)

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


@register
class Qwen35(Arch):
    """Qwen 3.5 / 3.8: hybrid DeltaNet-attention with gated outputs.

    Three layers out of four are Gated DeltaNet recurrent blocks; the
    fourth is full attention with a learned sigmoid gate on the output.
    Both types share a standard SwiGLU feed-forward after the mixing.

    The DeltaNet block carries a per-head outer-product state matrix
    rather than a KV cache, so memory per token is zero for those
    layers. The state is [n_v_heads, head_dim, head_dim] updated by
    ``S = exp(gate) * S + beta * outer(k, v)`` each token.

    The full-attention block projects Q to twice the head dimension:
    the first half is the query (normalised), the second half is a
    sigmoid gate applied to the attention output.
    """

    name = "qwen35"
    rope_style = "neox"
    pretokenizer = "qwen35"
    hf_names = (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "qwen3_5",
        "qwen3_5_text",
    )

    # HF tensor name fragments → GGUF equivalents, applied after the
    # ``language_model.model.layers.N.`` prefix is stripped to ``blk.N.``.
    _LAYER_MAP = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "post_attention_norm.weight",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
        # SSM / recurrent layers
        "linear_attn.in_proj_qkv": "attn_qkv",
        "linear_attn.in_proj_z": "attn_gate",
        "linear_attn.in_proj_a": "ssm_alpha",
        "linear_attn.in_proj_b": "ssm_beta",
        "linear_attn.conv1d.weight": "ssm_conv1d.weight",
        "linear_attn.A_log": "ssm_a",
        "linear_attn.dt_bias": "ssm_dt.bias",
        "linear_attn.norm.weight": "ssm_norm.weight",
        "linear_attn.out_proj": "ssm_out",
        # Full-attention layers
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
    }

    _GLOBAL_MAP = {
        "language_model.model.embed_tokens": "token_embd",
        "language_model.model.norm.weight": "output_norm.weight",
        "language_model.lm_head": "output",
    }

    def translate_name(self, hf_name: str) -> str | None:
        import re

        # Global tensors (embedding, final norm, output head).
        for hf_prefix, gguf_name in self._GLOBAL_MAP.items():
            if hf_name.startswith(hf_prefix):
                suffix = hf_name[len(hf_prefix):]
                if suffix and suffix.startswith("."):
                    return gguf_name + suffix
                return gguf_name

        # Per-layer tensors.
        m = re.match(
            r"language_model\.model\.layers\.(\d+)\.(.*)", hf_name
        )
        if not m:
            return None
        layer, rest = m.group(1), m.group(2)
        prefix = f"blk.{layer}."

        # Try longest match first so ``mlp.gate_proj.weight`` matches
        # ``mlp.gate_proj`` (→ ``ffn_gate``) + ``.weight`` suffix,
        # not a shorter prefix.
        for hf_frag, gguf_frag in sorted(
            self._LAYER_MAP.items(), key=lambda kv: -len(kv[0])
        ):
            if rest.startswith(hf_frag):
                tail = rest[len(hf_frag):]
                if tail and tail.startswith("."):
                    return prefix + gguf_frag + tail
                return prefix + gguf_frag

        return None

    def translate_config(self, config: dict) -> list[tuple[str, str, str]]:
        tc = config.get("text_config") or config
        a = self.name
        rows = []

        def add(key, value):
            if value is not None:
                rows.append((f"{a}.{key}", str(value), "string"))

        n_layers = tc.get("num_hidden_layers", 0)
        mtp = tc.get("mtp_num_hidden_layers", 0)
        add("block_count", n_layers + mtp)
        add("nextn_predict_layers", mtp)
        add("embedding_length", tc.get("hidden_size"))
        add("feed_forward_length", tc.get("intermediate_size"))
        add("attention.head_count", tc.get("num_attention_heads"))
        add("attention.head_count_kv", tc.get("num_key_value_heads"))
        add("attention.key_length", tc.get("head_dim"))
        add("attention.layer_norm_rms_epsilon", tc.get("rms_norm_eps"))
        add("context_length", tc.get("max_position_embeddings"))

        rope = tc.get("rope_parameters") or {}
        add("rope.freq_base", rope.get("rope_theta"))

        head_dim = tc.get("head_dim", 256)
        partial = tc.get("partial_rotary_factor") or rope.get(
            "partial_rotary_factor", 1.0
        )
        add("rope.dimension_count", int(head_dim * partial))

        add("ssm.conv_kernel", tc.get("linear_conv_kernel_dim"))
        add("ssm.state_size", tc.get("linear_key_head_dim"))
        add("ssm.group_count", tc.get("linear_num_key_heads"))
        add("ssm.time_step_rank", tc.get("linear_num_value_heads"))
        n_v = tc.get("linear_num_value_heads")
        k_hd = tc.get("linear_key_head_dim")
        if n_v is not None and k_hd is not None:
            add("ssm.inner_size", n_v * k_hd)
        add("full_attention_interval", tc.get("full_attention_interval"))

        return rows

    # The fused recurrent projection arrives in the order `_ssm_block`
    # already reads it -- one run of queries, one of keys, one of values.
    # A safetensors export was checked against the GGUF build of this same
    # architecture, whose order is known good, by comparing the weight
    # magnitude of the three segments: the ratios agree to 7e-04, where
    # regrouping the rows by key-head instead moves them by 0.16. So no
    # reordering happens here, and this note is what says that was measured
    # rather than assumed.

    def needs_transform(self, name: str) -> bool:
        return name.endswith(".ssm_a") or name.endswith(".ssm_conv1d.weight")

    def transform_tensor(self, name: str, data: np.ndarray) -> np.ndarray:
        if name.endswith(".ssm_a"):
            return -np.exp(data.astype(np.float32)).astype(data.dtype)
        if name.endswith(".ssm_conv1d.weight") and 1 in data.shape:
            return data.squeeze()
        return data

    def configure(self, cfg, meta: dict, store) -> None:
        a = cfg.arch

        def num(key, default=None):
            value = meta.get(f"{a}.{key}")
            if value is None:
                return default
            return (float(value) if "." in str(value) or "e" in str(value).lower()
                    else int(value))

        # `block_count` counts one more block than the model runs. The last
        # is the multi-token-prediction head -- it carries `nextn.*` tensors
        # and no attention or recurrent mixing of its own, and it exists to
        # let a speculative decoder guess the token after next. Running it as
        # if it were layer 65 of the stack reads tensors that are not there.
        # Generation uses the layers below it and stops.
        cfg.nextn_layers = int(num("nextn_predict_layers", 0))
        cfg.n_layers -= cfg.nextn_layers

        cfg.ssm_conv_kernel = int(num("ssm.conv_kernel", 4))
        cfg.ssm_state_size = int(num("ssm.state_size", 128))
        cfg.ssm_group_count = int(num("ssm.group_count", 16))
        cfg.ssm_dt_rank = int(num("ssm.time_step_rank", 48))
        cfg.ssm_inner_size = int(num("ssm.inner_size", 6144))
        cfg.full_attn_interval = int(num("full_attention_interval", 4))
        # The recurrent block's head geometry is not the attention block's,
        # and every number below is pinned by a tensor shape rather than
        # guessed: ssm_norm is [state_size], so that is the head width for
        # both keys and values; inner_size / that width gives the value
        # heads, which is why it equals time_step_rank -- the decay and the
        # input gate carry one value per value head. The keys have
        # group_count heads of the same width, so 2048 = 16 * 128 is the
        # key half of the fused projection, twice over for query and key.
        cfg.ssm_head_dim = cfg.ssm_state_size
        cfg.ssm_n_v_heads = cfg.ssm_inner_size // cfg.ssm_head_dim
        cfg.ssm_n_k_heads = cfg.ssm_group_count
        cfg.ssm_k_dim = cfg.ssm_n_k_heads * cfg.ssm_head_dim
        cfg.ssm_v_dim = cfg.ssm_n_v_heads * cfg.ssm_head_dim

        # The attention scale is left as the shared config worked it out,
        # 1/sqrt(head_dim). Normalising queries and keys before the product
        # does not replace it here: this architecture keeps both, where
        # gemma folds the scale into its query norm and drops it. Setting it
        # to 1 with a head width of 256 makes every score sixteen times too
        # large, which saturates the softmax into attending to one token and
        # reads as a model that has lost the thread rather than as an error.

        # One layer in `full_attention_interval` attends; the rest recur.
        cfg.is_ssm_layer = [
            (i + 1) % cfg.full_attn_interval != 0
            for i in range(cfg.n_layers)
        ]

        # The recurrence multiplies its state by exp(softplus(...) * ssm_a)
        # every token, and softplus is positive, so ssm_a carries the sign.
        # It has to be negative: at zero the state never forgets, and above
        # zero it grows without bound. Neither raises anywhere on its own --
        # the state saturates and the model produces fluent nonsense, which
        # is the failure this whole file is written to avoid.
        #
        # The reference computes the log-decay as
        #     g = -exp(A_log) * softplus(a + dt_bias)
        # so the checkpoint's parameter is a *log*, positive, and the
        # negation and exponential are applied around it. Converters fold
        # both in and write -exp(A_log) directly, which is why multiplying
        # by the stored value is right and the stored value is negative. A
        # file that kept the raw A_log instead would need `-exp(ssm_a)`
        # here, and this is what says so rather than leaving it to be
        # inferred from bad output.
        if store.has("blk.0.ssm_a"):
            worst = float(store.get_numpy("blk.0.ssm_a").max())
            if worst >= 0.0:
                raise ValueError(
                    f"This model's ssm_a is not negative (max {worst:+.4g}), "
                    f"so the recurrent state would not decay and the model "
                    f"would produce fluent nonsense rather than fail.\n"
                    f"reminis reads the decay as "
                    f"exp(softplus(alpha + dt_bias) * ssm_a), which expects "
                    f"the converter to have folded in the reference's "
                    f"leading minus and exponential. This file appears to "
                    f"store the raw A_log, which needs -exp(ssm_a) instead."
                )

        # The key/value cache advances its length on the last layer, on the
        # assumption that every layer appended to it. Here most layers do
        # not, so that only holds while the last one attends -- which it
        # does for this architecture as published. If a future variant
        # ends on a recurrent layer, the cache would silently stop growing
        # and every token after the first would attend to a stale span, so
        # this refuses rather than generating quietly wrong text.
        if cfg.is_ssm_layer[-1]:
            raise ValueError(
                f"This model's last layer is recurrent, which the key/value "
                f"cache does not support: it tracks its length on layer "
                f"{cfg.n_layers - 1}, and that layer never appends to it."
            )

    def _causal_conv1d(self, model, x, kernel, conv_state):
        """A short depthwise convolution over time, carrying its own state.

        `x` is (tokens, channels) and the kernel is one filter per channel.
        GGUF writes it (kernel_size, channels) and reminis reverses shapes on
        the way out of the database, so it arrives transposed and is put back
        here rather than at every call site.

        The state is the last `kernel_size - 1` inputs, which is what makes
        the convolution causal across calls as well as within one: decoding
        hands in a single token and the window it needs is the three that
        came before, which were seen on earlier calls and are gone otherwise.
        """
        b = model.backend
        xp = b.xp
        channels = x.shape[-1]
        if kernel.shape[0] == channels:
            kernel = kernel.T
        k_size = kernel.shape[0]

        if conv_state is None:
            conv_state = b.zeros((k_size - 1, channels))
        padded = xp.concatenate([conv_state, x], axis=0)

        # One term per kernel tap rather than one per token: tap j multiplies
        # the whole sequence shifted by j, so a prompt of any length costs
        # four vector operations instead of four per token. The two are the
        # same sum, reassociated.
        n = x.shape[0]
        out = padded[0:n] * kernel[0]
        for j in range(1, k_size):
            out = out + padded[j:j + n] * kernel[j]

        # The carried window has the same chaining problem the recurrent
        # state does -- it is a slice of an array built from the previous
        # call's window. Both are forced together by the caller, once per
        # layer, rather than separately here and there.
        return out, padded[-(k_size - 1):]

    def _l2_norm(self, backend, x, eps=1e-6):
        """Queries and keys are unit vectors before the recurrence.

        The epsilon matches the reference implementation's `l2norm(..., eps
        =1e-6)` rather than being chosen for float32 -- it is large enough
        to matter on a short head, so a smaller one is a different function.
        """
        xp = backend.xp
        return x / xp.sqrt(xp.sum(x * x, axis=-1, keepdims=True) + eps)

    # Compiling this step was tried and removed. `mx.compile` over the whole
    # recurrence -- the obvious move, since it is a dozen small elementwise
    # operations issued one at a time, forty-eight layers deep on the 27B --
    # measured 25.29 tok/s against 25.07 on the 4B and 4.36 against 4.6 on
    # the 27B. Neither is a difference. What the step spends is evidently in
    # the two matrix products, which were already single kernels, and not in
    # the dispatch around them. It is recorded here so the next person does
    # not spend the afternoon finding out the same thing.

    def _deltanet_scan(self, model, q, k, v, gate, beta, ssm_state):
        """Gated DeltaNet recurrence, one step per token, all heads at once.

        q, k: [n_tokens, n_k_heads, head_dim]
        v:    [n_tokens, n_v_heads, head_dim]
        gate: [n_tokens, n_v_heads] -- exp(gate) is the decay
        beta: [n_tokens, n_v_heads] -- the input gate

        The state is one matrix per value head, [key_dim, value_dim], and a
        token both writes to it and reads from it. The update is a *delta*
        rule, which is what the name of the architecture is about and is
        not the same as gated linear attention:

            S      = exp(gate) * S          decay what is already stored
            stored = S^T k                  what S currently returns for k
            S      = S + k (v - stored)^T   write only the difference
            y      = S^T q

        The third line is the whole point. Writing `S + beta * k v^T`
        instead -- accumulating the value rather than the correction --
        looks nearly the same and behaves nothing like it: a key that
        recurs keeps adding its value to a direction that already holds it,
        the state grows along that direction until it dominates every
        readout, and the model emits the same token forever. It does not
        diverge to infinity or produce a NaN, so nothing catches it; the
        output is simply stuck. That was the first thing this
        implementation did on the real weights.

        The loop over positions is inherent -- each state depends on the one
        before it -- but the loop over heads is not, and running 48 of them
        as separate small matrix products per token per layer costs far more
        in dispatch than in arithmetic. So heads are a batch axis here and
        only time is stepped.

        Keys and queries come in grouped, `n_v_heads / n_k_heads` value heads
        to each, the same arrangement grouped-query attention uses.

        The state is held in float32 even where the rest of the model is
        half precision, which is what the reference does and is not a
        precaution to be economised on. Every other tensor here is written
        once and read once, so a rounding is a rounding; this one is fed
        back into itself at every token, so a rounding compounds. With a
        decay near one a head accumulates hundreds of contributions before
        the oldest fades, and float16 has about three decimal digits to
        hold the sum in. The error would not look like an error -- it looks
        like a model that gradually loses the thread of a long answer.
        """
        b = model.backend
        xp = b.xp
        n_tokens, n_v_heads, head_dim = v.shape
        n_k_heads = q.shape[1]
        if n_v_heads > n_k_heads:
            # Value head h uses key head h % n_k_heads -- the key heads are
            # cycled through, not held while the value heads advance. The
            # difference is exactly repeat versus tile, both produce an
            # array of the right shape, and only one pairs each value with
            # the key its weights were trained against.
            times = n_v_heads // n_k_heads
            q = xp.concatenate([q] * times, axis=1)
            k = xp.concatenate([k] * times, axis=1)

        wide = b.to_compute32
        q, k, v = wide(q), wide(k), wide(v)
        if ssm_state is None:
            ssm_state = xp.zeros((n_v_heads, head_dim, head_dim),
                                 dtype=type(b).float32_dtype())

        decay = xp.exp(wide(gate))
        beta = wide(beta)
        # The readout is scaled by 1/sqrt(head_dim); the state is not. This
        # is the one part of the recurrence that the architecture's own
        # description does not mention and only the kernel shows -- without
        # it every value leaving a recurrent layer is eleven times too
        # large here, which is not enough to overflow and is far too much
        # to be right.
        scale = 1.0 / math.sqrt(head_dim)
        outputs = []
        for t in range(n_tokens):
            kt = k[t][:, :, None]                       # (heads, key, 1)
            ssm_state = decay[t][:, None, None] * ssm_state

            # What the state already returns for this key, and the part of
            # the new value it does not yet account for. Reading is the
            # state contracted over its key axis, which is a matrix product
            # against the transpose rather than an einsum.
            stored = (xp.swapaxes(ssm_state, -1, -2) @ kt)[..., 0]
            delta = (v[t] - stored) * beta[t][:, None]

            ssm_state = ssm_state + kt * delta[:, None, :]
            outputs.append(
                (xp.swapaxes(ssm_state, -1, -2) @ q[t][:, :, None])
                .reshape(1, n_v_heads * head_dim) * scale
            )

        # Back to the compute dtype on the way out: what follows is a
        # quantized matrix multiply, which wants the backend's own width.
        out = xp.concatenate(outputs, axis=0).astype(b.compute_dtype)

        return out, ssm_state

    def _ssm_block(self, model, x, layer, ssm_states, offset):
        b = model.backend
        xp = b.xp
        cfg = model.cfg
        p = f"blk.{layer}."

        h = b.rms_norm(x, model.store.get(p + "attn_norm.weight").reshape(-1),
                       cfg.rms_eps)

        qkv = model._linear(h, p + "attn_qkv.weight")
        z = model._linear(h, p + "attn_gate.weight")

        conv_kernel = model.store.get(p + "ssm_conv1d.weight")
        conv_state = ssm_states.get_conv(layer)
        qkv_conv, new_conv = self._causal_conv1d(model, qkv, conv_kernel,
                                                  conv_state)
        ssm_states.set_conv(layer, new_conv)
        qkv_conv = b.silu(qkv_conv)

        # Three contiguous blocks, not the per-group interleaving the
        # reference checkpoint uses: the conversion to GGUF already
        # rearranges `fix_query_key_value_ordering`'s grouped layout into
        # one run of queries, one of keys and one of values. Reading it the
        # reference's way instead was measurably worse, which is the only
        # evidence available -- the widths are consistent with both.
        n_tokens = x.shape[0]
        d = cfg.ssm_head_dim
        k_dim = cfg.ssm_k_dim

        q_part = self._l2_norm(b, qkv_conv[:, :k_dim].reshape(
            n_tokens, cfg.ssm_n_k_heads, d))
        k_part = self._l2_norm(b, qkv_conv[:, k_dim:2 * k_dim].reshape(
            n_tokens, cfg.ssm_n_k_heads, d))
        v_part = qkv_conv[:, 2 * k_dim:].reshape(
            n_tokens, cfg.ssm_n_v_heads, d)

        # The decay and the input gate carry one value per value head, so
        # both projections land at exactly n_v_heads and nothing is
        # broadcast: ssm_alpha and ssm_beta are [d_model, 48] and there are
        # 48 value heads.
        # In float32, as the reference computes it: this feeds an
        # exponential whose result multiplies the state at every token, so
        # it is the one scalar per head where half precision is visible.
        alpha = b.to_compute32(model._linear(h, p + "ssm_alpha.weight"))
        alpha = alpha + b.to_compute32(
            model.store.get(p + "ssm_dt.bias").reshape(1, -1))
        gate = _softplus(xp, alpha) * b.to_compute32(
            model.store.get(p + "ssm_a").reshape(1, -1))
        beta_val = b.sigmoid(model._linear(h, p + "ssm_beta.weight"))

        scan_state = ssm_states.get_ssm(layer)
        scan_out, new_state = self._deltanet_scan(
            model, q_part, k_part, v_part, gate, beta_val, scan_state)
        ssm_states.set_ssm(layer, new_state)

        # Collapse both carried states, once for the layer.
        #
        # Each is built from its own value at the previous token, so on a
        # lazy backend nothing collapses the chain: the logits are the only
        # thing evaluated per token, and by then the states have been
        # stored away and are no longer part of what the logits reference.
        # Left alone the graph grows without bound -- measured on this
        # model, memory climbed past the device's working set and every
        # token cost more than the one before, which reads as a model that
        # slows down the longer it talks rather than as a leak.
        #
        # Scheduled rather than waited on. Nothing in this layer or the
        # next needs the state's value -- it is read at the *following*
        # token -- so blocking here would stall the processor against the
        # device once per recurrent layer, forty-eight times a token on the
        # larger model, purely to learn something no one is asking. The
        # graph is released either way; the forward pass's own eval of the
        # logits is the one place a barrier is actually wanted.
        b.eval_async(scan_out, new_state, new_conv)

        ssm_norm = model.store.get(p + "ssm_norm.weight").reshape(-1)
        # Gated RMS norm: normalise per head, then scale by the gate. The
        # gate goes through silu rather than the sigmoid a gate usually
        # takes -- checked against the reference, whose norm is configured
        # with activation "silu". A sigmoid here still produces text, and
        # slightly wrong text is the hardest kind of wrong to notice.
        scan_out = _group_rms_norm(b, scan_out, ssm_norm,
                                   cfg.ssm_n_v_heads, cfg.ssm_head_dim,
                                   cfg.rms_eps)
        scan_out = scan_out * b.silu(z)

        attn_out = model._linear(scan_out, p + "ssm_out.weight")
        return x + attn_out

    def _attn_block(self, model, x, layer, cache, offset):
        b = model.backend
        xp = b.xp
        cfg = model.cfg
        p = f"blk.{layer}."
        n_tokens = x.shape[0]

        h = b.rms_norm(x, model.store.get(p + "attn_norm.weight").reshape(-1),
                       cfg.rms_eps)

        # The query projection is twice the head width, and the second half
        # of each *head* is a gate on that head's output -- not the second
        # half of the whole vector. The reference views the projection as
        # (heads, 2 * head_dim) before splitting, so flattened it reads
        # q0 g0 q1 g1 ... rather than q0 q1 ... g0 g1 ...
        #
        # Splitting the flat vector down the middle instead takes the first
        # half of the heads, queries and gates interleaved, and calls it the
        # query. Every head then attends with another head's gate applied to
        # something that is not a query. It produces confident nonsense and
        # nothing about the shapes objects, since both halves are the same
        # size either way.
        q_full = model._linear(h, p + "attn_q.weight").reshape(
            n_tokens, cfg.n_heads, 2 * cfg.head_dim)
        q = q_full[..., :cfg.head_dim]
        q_gate = q_full[..., cfg.head_dim:].reshape(n_tokens, -1)

        k = model._linear(h, p + "attn_k.weight")
        v = model._linear(h, p + "attn_v.weight")

        k = k.reshape(n_tokens, cfg.n_kv_heads, cfg.head_dim)
        v = v.reshape(n_tokens, cfg.n_kv_heads, cfg.head_dim)

        q = b.rms_norm(q, model.store.get(p + "attn_q_norm.weight").reshape(-1),
                       cfg.rms_eps)
        k = b.rms_norm(k, model.store.get(p + "attn_k_norm.weight").reshape(-1),
                       cfg.rms_eps)

        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]

        q = b.rope(q, cfg.rope_dim, False, cfg.rope_base, offset, cfg.rope_freqs)
        k = b.rope(k, cfg.rope_dim, False, cfg.rope_base, offset, cfg.rope_freqs)

        k_all, v_all = cache.append(layer, k, v)

        mask = None
        if n_tokens > 1:
            mask = model._causal_mask(n_tokens, offset, k_all.shape[-2], 0)

        out = b.attention(q, k_all, v_all, cfg.attn_scale, mask, None)
        out = out[0].transpose(1, 0, 2).reshape(n_tokens, cfg.n_heads * cfg.head_dim)
        out = out * b.sigmoid(q_gate)

        return x + model._linear(out, p + "attn_output.weight")

    def _states(self, model):
        states = getattr(model, "_ssm_states", None)
        if states is None:
            states = SSMState(model.cfg.n_layers, model.backend)
            model._ssm_states = states
        return states

    def snapshot_state(self, model):
        """Three quarters of this model's layers are recurrent, so a
        forward pass cannot be undone by truncating the key/value cache
        alone -- the DeltaNet state and the convolution window have
        already absorbed the tokens."""
        return self._states(model).snapshot()

    def restore_state(self, model, snapshot) -> None:
        self._states(model).restore(snapshot)

    def block(self, model, x, layer: int, cache, offset: int):
        cfg = model.cfg
        p = f"blk.{layer}."

        ssm_states = self._states(model)

        if cfg.is_ssm_layer[layer]:
            x = self._ssm_block(model, x, layer, ssm_states, offset)
        else:
            x = self._attn_block(model, x, layer, cache, offset)

        b = model.backend
        ffn_norm_name = (p + "ffn_norm.weight" if model.store.has(p + "ffn_norm.weight")
                         else p + "post_attention_norm.weight")
        h = b.rms_norm(x, model.store.get(ffn_norm_name).reshape(-1), cfg.rms_eps)
        gate = model._linear(h, p + "ffn_gate.weight")
        up = model._linear(h, p + "ffn_up.weight")
        return x + model._linear(b.silu(gate) * up, p + "ffn_down.weight")


@register
class Lfm2(Arch):
    """Liquid Foundation Model 2: hybrid conv-attention with gated short convolutions.

    Two kinds of layer, specified by config.layer_types:

    **Conv layers** project the input to 3x the model width (B, C, x),
    element-multiply B*x, run a depthwise causal conv1d (kernel=3), gate
    the result by C, and project back. The gating makes this a multiplicative
    filter rather than a plain convolution, which is how a 3-wide kernel
    manages to carry information across a sequence.

    **Attention layers** are standard grouped-query attention with RMS-normed
    queries and keys, RoPE, and no output gate.

    Both types share a SwiGLU feed-forward (w1, w2, w3) after the mixing.

    The model norms the embedding after the stack rather than before the
    output projection -- the weight is called `embedding_norm` rather than
    the usual `model.norm`. Embeddings are tied.
    """

    name = "lfm2"
    rope_style = "neox"
    pretokenizer = "llama-bpe"
    hf_names = ("Lfm2ForCausalLM", "lfm2")

    _LAYER_MAP = {
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.out_proj": "attn_output",
        "self_attn.q_layernorm": "attn_q_norm",
        "self_attn.k_layernorm": "attn_k_norm",
        "conv.in_proj": "conv_in_proj",
        "conv.out_proj": "conv_out_proj",
        "conv.conv": "conv_conv",
        "feed_forward.w1": "ffn_gate",
        "feed_forward.w3": "ffn_up",
        "feed_forward.w2": "ffn_down",
        "operator_norm": "attn_norm",
        "ffn_norm": "ffn_norm",
    }

    _GLOBAL_MAP = {
        "model.embed_tokens": "token_embd",
        "model.embedding_norm": "output_norm",
    }

    def prepare_meta(self, meta: dict) -> dict:
        out = {}
        kv = meta.get(f"{self.name}.attention.head_count_kv")
        if kv is not None:
            values = _as_list(kv)
            nonzero = [int(v) for v in values if int(v) > 0]
            if nonzero:
                out[f"{self.name}.attention.head_count_kv"] = max(nonzero)
        return out

    def translate_name(self, hf_name: str) -> str | None:
        for hf_prefix, gguf_name in self._GLOBAL_MAP.items():
            if hf_name == hf_prefix or hf_name.startswith(hf_prefix + "."):
                return gguf_name + hf_name[len(hf_prefix):]

        m = re.match(r"model\.layers\.(\d+)\.(.*)", hf_name)
        if not m:
            return None
        layer, rest = m.group(1), m.group(2)
        prefix = f"blk.{layer}."

        for hf_frag, gguf_frag in sorted(
            self._LAYER_MAP.items(), key=lambda kv: -len(kv[0])
        ):
            if rest == hf_frag or rest.startswith(hf_frag + "."):
                return prefix + gguf_frag + rest[len(hf_frag):]
        return None

    def translate_config(self, config: dict) -> list[tuple[str, str, str]]:
        tc = config.get("text_config") or config
        a = self.name
        rows = []

        def add(key, value):
            if value is not None:
                rows.append((f"{a}.{key}", str(value), "string"))

        add("block_count", tc.get("num_hidden_layers"))
        add("embedding_length", tc.get("hidden_size") or tc.get("block_dim"))
        add("feed_forward_length", tc.get("intermediate_size"))
        add("attention.head_count", tc.get("num_attention_heads") or tc.get("num_heads"))
        add("attention.head_count_kv", tc.get("num_key_value_heads"))
        add("attention.layer_norm_rms_epsilon", tc.get("norm_eps") or tc.get("block_norm_eps"))
        add("context_length", tc.get("max_position_embeddings"))

        rope = tc.get("rope_parameters") or {}
        add("rope.freq_base", tc.get("rope_theta", rope.get("rope_theta")))

        layer_types = tc.get("layer_types", [])
        add("layer_types", json.dumps(layer_types))
        add("conv_L_cache", tc.get("conv_L_cache", 3))

        return rows

    def configure(self, cfg, meta: dict, store) -> None:
        a = cfg.arch

        def num(key, default=None):
            value = meta.get(f"{a}.{key}")
            if value is None:
                return default
            return (float(value) if "." in str(value) or "e" in str(value).lower()
                    else int(value))

        layer_types_raw = meta.get(f"{a}.layer_types")
        if layer_types_raw:
            layer_types = json.loads(layer_types_raw) if isinstance(layer_types_raw, str) else layer_types_raw
        else:
            kv_raw = meta.get(f"{a}.attention.head_count_kv", "[]")
            kv_per_layer = _as_list(kv_raw)
            layer_types = [
                "full_attention" if int(v) > 0 else "conv"
                for v in kv_per_layer
            ]
        cfg.lfm2_layer_types = layer_types
        cfg.lfm2_is_conv = [t != "full_attention" for t in layer_types]
        conv_kernel = num("conv_L_cache") or num("shortconv.l_cache") or 3
        cfg.lfm2_conv_kernel = int(conv_kernel)

        if all(cfg.lfm2_is_conv):
            raise ValueError(
                "This model has no attention layers, so there is nothing "
                "to populate the key/value cache."
            )

        # The KV cache advances its length counter on the last layer that
        # calls append. For a hybrid model the last layer may be a conv that
        # never appends, so point the cache at the last attention layer.
        last_attn = max(i for i, is_conv in enumerate(cfg.lfm2_is_conv) if not is_conv)
        cfg.last_kv_layer = last_attn

    def _conv_layer(self, model, layer):
        """Names, norm and the conv kernel for one conv layer, resolved once.

        The kernel arrives as (channels, 1, k) or (k, 1, channels) depending
        on which converter wrote it, and the block wants it as k rows of
        channel-wide weights. Squeezing and transposing that per token per
        layer was 20 reshapes a token that always produced the same array.
        """
        cache = getattr(model, "_lfm2_conv_cache", None)
        if cache is None:
            cache = {}
            model._lfm2_conv_cache = cache
        entry = cache.get(layer)
        if entry is not None:
            return entry

        store = model.store
        p = f"blk.{layer}."
        if store.has(p + "conv_in_proj.weight"):
            in_w = p + "conv_in_proj.weight"
            conv_w = p + "conv_conv.weight"
            out_w = p + "conv_out_proj.weight"
        else:
            in_w = p + "shortconv.in_proj.weight"
            conv_w = p + "shortconv.conv.weight"
            out_w = p + "shortconv.out_proj.weight"

        kernel = store.get(conv_w).squeeze()
        if kernel.ndim == 1:
            kernel = kernel.reshape(-1, 1)
        # Rows must be taps, columns channels. A square kernel is ambiguous
        # and does not occur -- k is 3 and channels is thousands.
        if kernel.shape[0] == model.cfg.d_model:
            kernel = kernel.T
        kernel = model.backend.contiguous(kernel)
        # The taps are read one at a time in the conv; slicing them once here
        # keeps that out of the per-token path.
        taps = [kernel[j] for j in range(kernel.shape[0])]

        norm = store.get(p + "attn_norm.weight").reshape(-1)
        entry = (in_w, out_w, norm, taps)
        if not store.stream:
            cache[layer] = entry
        return entry

    def _causal_conv1d(self, model, x, taps, conv_state):
        b = model.backend
        xp = b.xp
        k_size = len(taps)

        if conv_state is None:
            conv_state = b.zeros((k_size - 1, x.shape[-1]))
        padded = xp.concatenate([conv_state, x], axis=0)

        n = x.shape[0]
        out = padded[0:n] * taps[0]
        for j in range(1, k_size):
            out = out + padded[j:j + n] * taps[j]

        return out, padded[-(k_size - 1):]

    def _conv_block(self, model, x, layer, conv_states, offset):
        b = model.backend
        cfg = model.cfg
        in_w, out_w, norm, taps = self._conv_layer(model, layer)

        h = b.rms_norm(x, norm, cfg.rms_eps)
        proj = model._linear(h, in_w)

        third = proj.shape[-1] // 3
        gate_b = proj[:, :third]
        gate_c = proj[:, third:2 * third]
        value = proj[:, 2 * third:]

        conv_out, new_conv = self._causal_conv1d(
            model, gate_b * value, taps, conv_states.get_conv(layer))
        conv_states.set_conv(layer, new_conv)

        return x + model._linear(gate_c * conv_out, out_w)

    def _attn_layer(self, model, layer):
        """The three norms one attention layer reads, resolved once."""
        cache = getattr(model, "_lfm2_attn_cache", None)
        if cache is None:
            cache = {}
            model._lfm2_attn_cache = cache
        entry = cache.get(layer)
        if entry is not None:
            return entry

        store = model.store
        p = f"blk.{layer}."
        entry = (
            store.get(p + "attn_norm.weight").reshape(-1),
            store.get(p + "attn_q_norm.weight").reshape(-1),
            store.get(p + "attn_k_norm.weight").reshape(-1),
        )
        if not store.stream:
            cache[layer] = entry
        return entry

    def _attn_block(self, model, x, layer, cache, offset):
        b = model.backend
        cfg = model.cfg
        p = f"blk.{layer}."
        n_tokens = x.shape[0]
        attn_norm, q_norm, k_norm = self._attn_layer(model, layer)

        h = b.rms_norm(x, attn_norm, cfg.rms_eps)

        q = model._linear(h, p + "attn_q.weight")
        k = model._linear(h, p + "attn_k.weight")
        v = model._linear(h, p + "attn_v.weight")

        q = q.reshape(n_tokens, cfg.n_heads, cfg.head_dim)
        k = k.reshape(n_tokens, cfg.n_kv_heads, cfg.head_dim)
        v = v.reshape(n_tokens, cfg.n_kv_heads, cfg.head_dim)

        q = b.rms_norm(q, q_norm, cfg.rms_eps)
        k = b.rms_norm(k, k_norm, cfg.rms_eps)

        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]

        q = b.rope(q, cfg.rope_dim, False, cfg.rope_base, offset, cfg.rope_freqs)
        k = b.rope(k, cfg.rope_dim, False, cfg.rope_base, offset, cfg.rope_freqs)

        k_all, v_all = cache.append(layer, k, v)

        mask = None
        if n_tokens > 1:
            mask = model._causal_mask(n_tokens, offset, k_all.shape[-2], 0)

        out = b.attention(q, k_all, v_all, cfg.attn_scale, mask, None)
        out = out[0].transpose(1, 0, 2).reshape(n_tokens, cfg.n_heads * cfg.head_dim)

        return x + model._linear(out, p + "attn_output.weight")

    def _states(self, model):
        states = getattr(model, "_lfm2_conv_states", None)
        if states is None:
            states = _Lfm2ConvState(model.cfg.n_layers)
            model._lfm2_conv_states = states
        return states

    def snapshot_state(self, model):
        return self._states(model).snapshot()

    def restore_state(self, model, snapshot) -> None:
        self._states(model).restore(snapshot)

    def _ffn_norm(self, model, layer):
        cache = getattr(model, "_lfm2_ffn_cache", None)
        if cache is None:
            cache = {}
            model._lfm2_ffn_cache = cache
        norm = cache.get(layer)
        if norm is None:
            norm = model.store.get(f"blk.{layer}.ffn_norm.weight").reshape(-1)
            if not model.store.stream:
                cache[layer] = norm
        return norm

    def block(self, model, x, layer: int, cache, offset: int):
        cfg = model.cfg
        b = model.backend
        p = f"blk.{layer}."

        conv_states = self._states(model)

        if cfg.lfm2_is_conv[layer]:
            x = self._conv_block(model, x, layer, conv_states, offset)
        else:
            x = self._attn_block(model, x, layer, cache, offset)

        norm = self._ffn_norm(model, layer)
        h = b.rms_norm(x, norm, cfg.rms_eps)
        gate, up = model._gate_up(h, layer, p)
        return x + model._linear(b.silu(gate) * up, p + "ffn_down.weight")


class _Lfm2ConvState:
    """Conv state carried across tokens for Lfm2 conv layers."""

    def __init__(self, n_layers):
        self._conv = [None] * n_layers

    def snapshot(self):
        return list(self._conv)

    def restore(self, snapshot) -> None:
        self._conv = list(snapshot)

    def get_conv(self, layer):
        return self._conv[layer]

    def set_conv(self, layer, state):
        self._conv[layer] = state


class SSMState:
    """Hidden state for DeltaNet layers — conv buffers and scan states.

    Every update below replaces an entry rather than writing into one --
    the scan builds a new array each step and `set_ssm` swaps it in -- so
    a snapshot is a copy of the two lists and not of the arrays they hold.
    That is what makes rolling a rejected speculation back cheap: the old
    state stays alive because the snapshot still refers to it, and the new
    one is dropped when the snapshot is put back.
    """

    def __init__(self, n_layers, backend):
        self._conv = [None] * n_layers
        self._ssm = [None] * n_layers
        self.backend = backend

    def snapshot(self):
        return list(self._conv), list(self._ssm)

    def restore(self, snapshot) -> None:
        self._conv, self._ssm = list(snapshot[0]), list(snapshot[1])

    def get_conv(self, layer):
        return self._conv[layer]

    def set_conv(self, layer, state):
        self._conv[layer] = state

    def get_ssm(self, layer):
        return self._ssm[layer]

    def set_ssm(self, layer, state):
        self._ssm[layer] = state


def _softplus(xp, x):
    """log(1 + exp(x)), by a route that does not overflow.

    Written directly, the exponential is unbounded and half precision runs
    out at about x = 11: exp(12) is already infinity in float16, so
    log(1 + exp(x)) returns infinity for every larger input. That does not
    raise. It makes the decay exp(inf * negative) exactly zero, so the
    affected heads quietly forget everything and the model keeps talking.

    The identity below moves the large part outside the exponential --
    softplus(x) = max(x, 0) + log1p(exp(-|x|)) -- so the argument is never
    positive and the result is exact across the whole range.
    """
    return xp.maximum(x, 0) + xp.log1p(xp.exp(-xp.abs(x)))


def _group_rms_norm(backend, x, weight, n_heads, head_dim, eps):
    shape = x.shape
    x = x.reshape(*shape[:-1], n_heads, head_dim)
    x = backend.rms_norm(x, weight, eps)
    return x.reshape(shape)
