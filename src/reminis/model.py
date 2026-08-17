"""The forward pass itself.

One embedding lookup, a stack of blocks, one final norm, one projection to
the vocabulary. The arithmetic is written against a backend rather than
against numpy, so the same code runs on MLX and CuPy, and every place the
architectures differ -- the rotary layout, whether a layer's attention
slides, how a mixture routes -- is asked of the model's ``arch.py`` entry
rather than branched on here.

What is worth knowing before reading it:

  * Prefill and decode are the same function. ``forward`` takes a list of
    tokens and a cache offset, so a prompt is one call with many tokens
    and a step is one call with one.
  * A mixture-of-experts layer either gathers from a stacked tensor or
    fetches experts one at a time, depending on whether the database has
    an index over them. The index is what took gpt-oss-20b from 0.71 to
    37 tok/s.
  * Nothing here decides *how* a weight arrives in memory. That is
    ``weights.py``, and it is where the load time is.
"""

import os
import time

import numpy as np

from reminis.backend import select as select_backend
from reminis.config import Config
from reminis.errors import UnsupportedModel
from reminis.kvcache import KVCache
from reminis.tokenizer import build_tokenizer
from reminis.weights import WeightStore

# ---------------------------------------------------------------- the model


class Model:
    """A llama-family transformer whose weights come from SQLite.

    The arithmetic is written against a backend rather than against numpy
    directly, so the same forward pass runs on the CPU or on a GPU depending
    on what the machine has. numpy stays the reference: it is the one whose
    logits are checked against transformers to five decimal places, and the
    others are checked against it.
    """

    def __init__(self, db_path: str, stream: bool = False, backend=None,
                 pack_bits=None, pack_group: int = 128,
                 expert_cache: int = 0, expert_bits=None,
                 reader_threads: int = 0):
        self.backend = backend or select_backend("inference")
        self.store = WeightStore(db_path, stream=stream, backend=self.backend,
                                 pack_bits=pack_bits, pack_group=pack_group,
                                 reader_threads=reader_threads)
        if expert_cache:
            if not self.store.can_stream_experts():
                raise UnsupportedModel(
                    "Loading experts on demand needs incremental blob reads, "
                    "which arrived in Python 3.11. On 3.10 the whole expert "
                    "stack has to be read at once."
                )
            self.store.expert_cache_size = expert_cache
            # Left alone, a fetched expert is kept at the backend's compute
            # dtype, which is four times what the file holds but is the same
            # arithmetic the resident path does. Packing it down to 4 bits
            # fits four times as many in the same memory and is worth doing
            # -- but it changes the answer, so it is asked for rather than
            # assumed. An indexed model gets the width the index was built
            # at and ignores this entirely.
            self.store.expert_bits = expert_bits
        meta = dict(self.store.conn.execute("SELECT key, value FROM model_meta"))
        self.meta = meta
        self.cfg = Config(meta, self.store)
        self.tokenizer = build_tokenizer(meta)
        self._layer_cache: dict[int, tuple] = {}
        self._mask_cache: dict = {}
        self._fused_cache: dict[tuple[int, str], tuple] = {}
        if pack_bits is not None and not stream and self.backend.can_pack():
            self._wire_memory()
            self.store.start_prefetch(self._load_order())
            self._pack_widest_first()

    def _wire_memory(self):
        """Ask that the weights, once loaded, be allowed to stay put.

        On unified memory a packed weight is ordinary system memory, and
        the system compresses ordinary memory when it runs short. It does
        so silently, and the device then stalls faulting each weight back
        in on the way to using it -- which is the whole of the forward
        pass, every token.

        `preload_experts` has wired its index for this reason since it was
        written, with a measurement attached: a mixture model on this size
        of machine wandered between 4.6 and 33 tok/s left to the system's
        judgement and held a flat 41 tok/s once wired. A dense model was
        never given the same treatment, and it has the same problem for
        the same reason -- worse, if anything, because every weight is
        touched for every token rather than the few a router picks.

        The request is a ceiling rather than an allocation, so asking
        before the weights exist is the right time to ask: it is what
        stops them being compressed as they arrive. The runtime clamps it
        to what the device will actually hold.
        """
        budget = self.backend.memory_budget()
        if not budget:
            return
        try:
            size = os.path.getsize(self.store.path)
        except OSError:
            return
        # Packing lands near the file's own size, so that is the estimate.
        #
        # What must not be wired is everything else the load needs while it
        # is happening: the prefetched blocks, the float32 a quantization
        # is unpacked through, the copy the device narrows from it. Wiring
        # the weights *and* asking for room to build them leaves the system
        # nowhere to put anything, and it responds by compressing whatever
        # is not wired -- which is precisely those working buffers. Asking
        # for a quarter more than the file, measured on a 9.9 GB model on a
        # 16 GB machine, pinned 12.3 GB and left the loader thrashing at
        # half a processor with nothing to show for it.
        #
        # So the request is the estimate itself, held back from the whole
        # of what the device offers. A model that fits easily is unaffected
        # -- it is below the cap either way.
        keep_free = max(int(budget * 0.15), WeightStore.PREFETCH_HEADROOM)
        self.backend.reserve(min(size, max(budget - keep_free, 0)))

    def _load_order(self):
        """The tensors a forward pass will want, roughly in the order it
        wants them.

        Only a hint for the prefetcher: it decides how far ahead to read,
        and being wrong about the order costs a little overlap and nothing
        else. Embeddings first because `_pack_widest_first` takes them
        before anything, then each block's weights in the order a layer
        walks them.
        """
        names = [n for n in ("token_embd.weight", "output.weight")
                 if self.store.has(n)]
        for layer in range(self.cfg.n_layers):
            p = f"blk.{layer}."
            names.extend(
                p + suffix for suffix in (
                    "attn_qkv.weight", "attn_gate.weight",
                    "attn_q.weight", "attn_k.weight", "attn_v.weight",
                    "attn_output.weight", "ssm_out.weight",
                    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                )
                if self.store.has(p + suffix)
            )
        return names

    def _pack_widest_first(self):
        """Pack the two vocabulary-sized matrices before anything else.

        Packing a quantization with no affine form has to unpack it to
        float32 first, and for a matrix the width of the vocabulary that
        intermediate dwarfs the result: measured on this model's embedding,
        1.27 billion weights cost 8.5 GB at their peak and 0.52 GB once
        packed, a factor of sixteen.

        A forward pass reaches the embedding first and the output projection
        last, so left alone the second of those spikes lands when every
        layer is already resident -- 11 GB plus 8.5 GB on a 17 GB machine,
        which does not fail, it swaps, and a model that would have run
        merely becomes unusable. Paying both spikes up front, while nothing
        else is loaded, keeps the high-water mark where the packed model
        is rather than where its largest intermediate is.

        Nothing here changes what ends up in memory, only the order it
        arrives in, so a model with room to spare is unaffected.
        """
        for name in ("token_embd.weight", "output.weight"):
            if self.store.has(name):
                self.store.get(name)

    def close(self):
        self.store.close()

    # -- resident experts --------------------------------------------------

    # How much of the device's working set the expert index may claim. The
    # rest goes to the dense weights, the key/value cache and the
    # activations, which are wanted resident just as much and are not
    # counted here because they are not loaded yet.
    PRELOAD_SHARE = 0.8

    def preload_experts(self, progress=None) -> dict:
        """Read the whole expert index into memory and pin it there.

        Reading experts one at a time as the router asks for them is what
        makes a model larger than memory runnable at all, but it is not
        what makes it fast. When the index is small enough to hold, holding
        all of it turns every fetch into a cache hit and takes the disk out
        of the loop entirely.

        Three things have to happen, and leaving out any one of them gives
        back most of the gain:

        Read it. The index is contiguous and in the kernel's layout, so
        this is a sequential scan rather than 2,304 decodes.

        Touch it. A buffer that has been copied to the device has not yet
        been read by it, and the first read faults it in. Measured on
        gpt-oss-20b, a token whose experts had never been multiplied took
        96 ms against 9.5 ms once they had, so the decode rate climbed from
        1.4 to 24 tok/s over the first seventy tokens and never reached its
        ceiling. Multiplying each expert by a zero vector at load time is a
        few seconds and moves the whole curve to the front.

        Pin it. Unified memory is ordinary system memory, and the system
        will compress it under pressure. See `Backend.reserve`.
        """
        store = self.store
        if not store._expert_layouts:
            raise UnsupportedModel(
                "This database has no expert index to preload.\n"
                "Build one:  reminis prepare <db>"
            )
        b = self.backend
        total = sum(l.n_experts for l in store._expert_layouts.values())
        needed = sum(l.block_bytes * l.n_experts
                     for l in store._expert_layouts.values())

        # Overcommitting is not a slow path, it is a wrong one. Asking the
        # device to hold 10.75 GB of experts against a 12.7 GB working set
        # left no room for the dense weights and produced fluent-looking
        # nonsense -- a row of exclamation marks -- rather than an error.
        # So the fit is checked before anything is read.
        budget = b.memory_budget()
        room = int(budget * self.PRELOAD_SHARE) if budget else 0
        if room and needed > room:
            raise UnsupportedModel(
                f"The expert index is {needed / 1e9:.2f} GB and this device "
                f"will hold about {room / 1e9:.2f} GB of it alongside the "
                f"rest of the model.\n"
                f"Either run it a piece at a time:  --experts "
                f"{max(1, int(total * room / needed / 100) * 100)}\n"
                f"or rebuild the index smaller:     reminis prepare <db> "
                f"--bits 3"
            )
        store.expert_cache_size = max(store.expert_cache_size, total)

        # The values do not matter, only that the device reads every byte of
        # the weight. The width does matter: the down projection takes the
        # feed-forward width where the others take the model width, so the
        # probe is built per tensor rather than assumed.
        probes = {}
        started = time.time()
        pending = []
        done = 0
        for name, layout in store._expert_layouts.items():
            probe = probes.get(layout.cols)
            if probe is None:
                probe = probes[layout.cols] = b.zeros((1, layout.cols))
            for index in range(layout.n_experts):
                pending.append(b.matmul_weight(probe, store.expert(name, index)))
                done += 1
                if len(pending) >= 64:
                    b.eval(pending)
                    pending = []
            if progress:
                progress(done, total, name, time.time() - started)
        b.eval(pending)

        b.reserve(int(needed / self.PRELOAD_SHARE))
        return {"experts": total, "bytes": needed,
                "seconds": time.time() - started}

    # -- one layer --------------------------------------------------------

    def _linear(self, x, name: str, bias: str | None = None):
        y = self.backend.matmul_weight(x, self.store.get(name))
        if bias and self.store.has(bias):
            y = y + self.store.get(bias).reshape(-1)
        return y

    def _fused(self, layer: int, key: str, names: list[str]):
        """One weight matrix standing in for several stacked ones.

        Q, K and V are three separate matrix-vector products reading three
        separate regions of memory, and so are the FFN's gate and up. Stacking
        each group into a single matrix turns them into one call over one
        contiguous region. Decoding is memory-bound, so what this buys is not
        arithmetic but streaming: measured on the real shapes, the fused QKV
        is about a third faster than the three separate ones.

        The originals are dropped from the cache as the fused copy is built,
        so this costs no extra memory. It is skipped entirely in streaming
        mode, where caching anything would defeat the point.
        """
        cached = self._fused_cache.get((layer, key))
        if cached is not None:
            return cached

        xp = self.backend.xp
        prefix = f"blk.{layer}."
        weight = self.backend.contiguous(
            xp.concatenate([self.store.get(prefix + n) for n in names], axis=0)
        )
        biases = [prefix + n.replace(".weight", ".bias") for n in names]
        bias = None
        if all(self.store.has(b) for b in biases):
            bias = xp.concatenate([self.store.get(b).reshape(-1) for b in biases])

        for n in names:
            self.store.drop(prefix + n)
        entry = (weight, bias)
        self._fused_cache[(layer, key)] = entry
        return entry

    def _qkv(self, h: np.ndarray, layer: int, prefix: str):
        # Packed weights cannot be stacked into one matrix -- concatenating
        # them would mean unpacking, which is the thing being avoided -- so
        # fusion and packing are alternatives, not companions.
        if self.store.stream or self.store.pack_bits is not None:
            return (self._linear(h, prefix + "attn_q.weight", prefix + "attn_q.bias"),
                    self._linear(h, prefix + "attn_k.weight", prefix + "attn_k.bias"),
                    self._linear(h, prefix + "attn_v.weight", prefix + "attn_v.bias"))

        weight, bias = self._fused_cache.get((layer, "qkv")) or self._fused(
            layer, "qkv",
            ["attn_q.weight", "attn_k.weight", "attn_v.weight"],
        )
        out = h @ weight.T
        if bias is not None:
            out = out + bias
        cut_q = self.cfg.n_heads * self.cfg.head_dim
        cut_k = cut_q + self.cfg.n_kv_heads * self.cfg.head_dim
        return out[:, :cut_q], out[:, cut_q:cut_k], out[:, cut_k:]

    def _gate_up(self, h: np.ndarray, layer: int, prefix: str):
        if self.store.stream or self.store.pack_bits is not None:
            return (self._linear(h, prefix + "ffn_gate.weight"),
                    self._linear(h, prefix + "ffn_up.weight"))

        weight, _ = self._fused_cache.get((layer, "gate_up")) or self._fused(
            layer, "gate_up", ["ffn_gate.weight", "ffn_up.weight"])
        out = h @ weight.T
        half = out.shape[1] // 2
        return out[:, :half], out[:, half:]

    def _layer_weights(self, layer: int):
        """Every weight a layer needs, resolved once.

        Looking these up by name inside the loop meant building eight
        strings and hashing eight dictionary keys per layer per token --
        360 string operations a token on a 30-layer model, which is real
        time when a whole token is under six milliseconds. Streaming mode
        skips the table, since its entire point is to hold nothing.
        """
        cached = self._layer_cache.get(layer)
        if cached is not None:
            return cached

        p = f"blk.{layer}."
        # Some models call the norm before the feed-forward "ffn_norm" and
        # others "post_attention_norm" -- the same position in the block,
        # named for what comes before it rather than after.
        ffn_norm_name = (p + "ffn_norm.weight" if self.store.has(p + "ffn_norm.weight")
                         else p + "post_attention_norm.weight")
        entry = (
            self.store.get(p + "attn_norm.weight").reshape(-1),
            self.store.get(ffn_norm_name).reshape(-1),
            p,
        )
        if not self.store.stream:
            self._layer_cache[layer] = entry
        return entry

    def _block(self, x, layer: int, cache: "KVCache", offset: int):
        cfg = self.cfg
        # An architecture whose block is a different computation rather than
        # a variation on this one supplies its own.
        own = cfg.spec.block(self, x, layer, cache, offset)
        if own is not None:
            return own

        b = self.backend
        xp = b.xp
        attn_norm, ffn_norm, p = self._layer_weights(layer)
        n_tokens = x.shape[0]

        h = b.rms_norm(x, attn_norm, cfg.rms_eps)

        q, k, v = self._qkv(h, layer, p)
        q = q.reshape(n_tokens, cfg.n_heads, cfg.head_dim)
        k = k.reshape(n_tokens, cfg.n_kv_heads, cfg.head_dim)
        v = v.reshape(n_tokens, cfg.n_kv_heads, cfg.head_dim)

        # (batch, heads, tokens, dim) is the layout both backends' rotary
        # and attention kernels expect, so the transposes happen once here
        # rather than being undone and redone between the two.
        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]

        q = b.rope(q, cfg.rope_dim, cfg.rope_style == "norm", cfg.rope_base,
                   offset, cfg.rope_freqs)
        k = b.rope(k, cfg.rope_dim, cfg.rope_style == "norm", cfg.rope_base,
                   offset, cfg.rope_freqs)

        k_all, v_all = cache.append(layer, k, v)

        # A single token attends to the whole cache, so there is nothing for
        # a causal mask to hide and building one is pure waste. It is only
        # needed when several tokens are processed at once.
        window = cfg.sliding_window if self._layer_slides(layer) else 0
        mask = None
        if n_tokens > 1 or window:
            mask = self._causal_mask(n_tokens, offset, k_all.shape[-2], window)

        sinks = None
        if self.store.has(p + "attn_sinks.weight"):
            sinks = self.store.get(p + "attn_sinks.weight").reshape(-1)

        out = b.attention(q, k_all, v_all, cfg.attn_scale, mask, sinks)
        out = out[0].transpose(1, 0, 2).reshape(n_tokens, cfg.n_heads * cfg.head_dim)
        attn_out = self._linear(out, p + "attn_output.weight", p + "attn_output.bias")
        x = x + cfg.residual_scale * attn_out

        if cfg.is_moe:
            h = b.rms_norm(x, ffn_norm, cfg.rms_eps)
            return x + cfg.residual_scale * self._moe_ffn(h, p)

        if self.store.stream or self.store.pack_bits is not None:
            h = b.rms_norm(x, ffn_norm, cfg.rms_eps)
            gate, up = self._gate_up(h, layer, p)
            branch = self._linear(b.silu(gate) * up, p + "ffn_down.weight")
            return x + cfg.residual_scale * branch

        gate_up, _ = self._fused_cache.get((layer, "gate_up")) or self._fused(
            layer, "gate_up", ["ffn_gate.weight", "ffn_up.weight"])
        if cfg.residual_scale != 1.0:
            h = b.rms_norm(x, ffn_norm, cfg.rms_eps)
            gate, up = self._gate_up(h, layer, p)
            branch = self._linear(b.silu(gate) * up, p + "ffn_down.weight")
            return x + cfg.residual_scale * branch
        return b.fused_ffn(x, ffn_norm, gate_up,
                           self.store.get(p + "ffn_down.weight"), cfg.rms_eps)

    def _expert_proj(self, x, prefix, name, idx, k, n_rows):
        """One projection through the experts chosen for each row.

        Some mixtures carry a bias per expert as well as a weight, and it
        has to be gathered alongside -- each row takes the bias of the
        expert it was routed to, not a shared one.
        """
        b = self.backend
        out = b.gather_matmul(x, self.store.get(f"{prefix}{name}.weight"), idx, k)
        bias_name = f"{prefix}{name}.bias"
        if self.store.has(bias_name):
            bias = self.store.get(bias_name)
            out = out + b.xp.take(bias, idx.reshape(-1), axis=0).reshape(
                n_rows, k, -1
            )
        return out

    def _moe_on_demand(self, h, prefix, idx, k, n_tokens):
        """Run the chosen experts, fetching each from the database as needed.

        One expert at a time, so what has to be resident is a few matrices
        rather than the whole stack. That is the trade the database makes
        available and a memory-mapped file does not: the router says which
        rows are wanted before any of them are read.
        """
        b = self.backend
        chosen = np.asarray(b.to_numpy(idx)).astype(int).reshape(n_tokens, k)

        # Every read this layer needs is known here, so ask for all of them
        # before waiting on any. The gate and up projections are wanted
        # first and the down projection only after the activation, but it
        # is the same disk and the same trip, so it goes in the same batch.
        self.store.prefetch_experts(
            (f"{prefix}{part}.weight", int(e))
            for e in np.unique(chosen)
            for part in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
        )

        results = []
        for token in range(n_tokens):
            row = h[token:token + 1]
            experts = [int(chosen[token, slot]) for slot in range(k)]

            # Both halves of the gate for every chosen expert are queued
            # before anything is looked at, so the eight matrix multiplies
            # are in flight together rather than one at a time.
            gates = [self._one_expert(row, prefix, "ffn_gate_exps", e)
                     for e in experts]
            ups = [self._one_expert(row, prefix, "ffn_up_exps", e)
                   for e in experts]
            hidden = [self._glu(g, u) for g, u in zip(gates, ups)]
            pieces = [self._one_expert(x, prefix, "ffn_down_exps", e)
                      for x, e in zip(hidden, experts)]
            results.append(b.xp.concatenate(pieces, axis=0))
        return b.xp.stack(results, axis=0)

    def _glu(self, gate, up):
        """The gated activation, in whichever form this model was trained on.

        Everything here is SwiGLU -- gate * sigmoid(gate) * up -- except
        gpt-oss, which clamps both sides, scales the sigmoid's input, and
        offsets the up projection by one.
        """
        b = self.backend
        cfg = self.cfg
        if not cfg.glu_limit:
            return b.silu(gate) * up
        xp = b.xp
        gate = xp.minimum(gate, cfg.glu_limit)
        up = xp.clip(up, -cfg.glu_limit, cfg.glu_limit)
        activated = gate * b.sigmoid(gate * cfg.glu_alpha)
        return (up + cfg.glu_up_offset) * activated

    def _one_expert(self, x, prefix, name, expert: int):
        """One projection through one expert, weight and bias alike.

        The bias tensors are small enough to hold whole -- one value per
        output, against a matrix per expert -- so only the weight is worth
        fetching a row at a time.
        """
        b = self.backend
        out = b.matmul_weight(x, self.store.expert(f"{prefix}{name}.weight", expert))
        bias_name = f"{prefix}{name}.bias"
        if self.store.has(bias_name):
            out = out + self.store.get(bias_name)[expert]
        return out

    def _moe_ffn(self, h, p: str):
        """A router picks a few experts per token; their outputs are summed.

        The experts are stored stacked -- one 3-D tensor per projection with
        the expert as its first axis -- so selecting them is a gather rather
        than a lookup of separate matrices. Every selected (token, expert)
        pair becomes a row of one batched matrix multiply, which is why this
        does not loop over experts in Python.
        """
        b = self.backend
        xp = b.xp
        cfg = self.cfg
        k = cfg.n_experts_used
        n_tokens = h.shape[0]

        router = self.store.get(p + "ffn_gate_inp.weight")
        logits = h @ router.T
        if self.store.has(p + "ffn_gate_inp.bias"):
            logits = logits + self.store.get(p + "ffn_gate_inp.bias").reshape(-1)
        probs = b.softmax(b.to_compute32(logits))

        # The k largest, then renormalised so the chosen weights sum to one.
        idx = xp.argpartition(-probs, k - 1, axis=-1)[:, :k]
        weight = xp.take_along_axis(probs, idx, axis=-1)
        weight = weight / xp.sum(weight, axis=-1, keepdims=True)

        if self.store.expert_cache_size:
            out = self._moe_on_demand(h, p, idx, k, n_tokens)
        else:
            gate = self._expert_proj(h, p, "ffn_gate_exps", idx, k, n_tokens)
            up = self._expert_proj(h, p, "ffn_up_exps", idx, k, n_tokens)
            hidden = (b.silu(gate) * up).reshape(n_tokens * k, -1)
            out = self._expert_proj(
                hidden, p, "ffn_down_exps", idx.reshape(-1)[:, None], 1,
                n_tokens * k,
            ).reshape(n_tokens, k, -1)
        return xp.sum(out * weight.astype(out.dtype)[..., None], axis=1)

    def _layer_slides(self, layer: int) -> bool:
        """Whether this layer sees only a window of the context.

        The pattern says how the layers alternate: a value of N means one
        layer in every N attends to everything and the rest are windowed,
        which is the convention llama.cpp records. Without a pattern, a
        stated window applies to every layer.
        """
        if not self.cfg.sliding_window:
            return False
        if self.cfg.swa_pattern <= 1:
            return True
        return (layer + 1) % self.cfg.swa_pattern != 0

    def _causal_mask(self, n_tokens: int, offset: int, total: int, window: int = 0):
        """True where a query at an absolute position may see a key.

        Built in numpy and handed to the backend, because it depends only on
        positions and is reused unchanged by every layer that shares a
        window size.
        """
        key = (n_tokens, offset, total, window)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        rows = np.arange(offset, offset + n_tokens)[:, None]
        cols = np.arange(total)[None, :]
        allowed = cols <= rows
        if window:
            allowed &= cols > rows - window
        mask = self.backend.xp.array(allowed)
        self._mask_cache[key] = mask
        return mask

    def forward(self, tokens: list[int], cache: "KVCache", offset: int,
                all_positions: bool = False) -> np.ndarray:
        """Logits for the last token of `tokens`, or for every one of them.

        Decoding only ever needs the last position, so that is the default and
        the output projection is applied to a single row. `all_positions` runs
        it across the whole batch instead, returning (n_tokens, vocab), which
        is what comparing two models position by position needs -- and doing it
        in one batched pass rather than one call per token is the difference
        between a matrix product and a few dozen matrix-vector products.

        Floating-point warnings are silenced for the duration. Two of them
        would otherwise fire on every token for reasons that are not bugs:
        the causal mask is built out of -inf on purpose, and Apple's
        Accelerate BLAS raises the divide-by-zero and overflow flags during
        perfectly ordinary float32 matmuls whose inputs and outputs are all
        finite. The logits are checked explicitly below instead, which
        catches a genuinely broken forward pass without the noise.
        """
        b = self.backend
        with b.errstate():
            embed = self.store.get("token_embd.weight")
            x = b.take_rows(embed, b.xp.array(np.asarray(tokens, dtype=np.int32)))
            if self.cfg.embedding_scale != 1.0:
                x = x * self.cfg.embedding_scale

            for layer in range(self.cfg.n_layers):
                x = self._block(x, layer, cache, offset)

            x = b.rms_norm(x, self.store.get("output_norm.weight").reshape(-1),
                           self.cfg.rms_eps)
            tail = x if all_positions else x[-1:]
            out_name = "token_embd.weight" if self.cfg.tied_output else "output.weight"
            logits = self.backend.matmul_weight(tail, self.store.get(out_name))
            if not all_positions:
                logits = logits.reshape(-1)
            if self.cfg.logit_scale != 1.0:
                logits = logits / self.cfg.logit_scale
            logits = self.cfg.spec.logits(self, logits)
            b.eval(logits)

        # Sampling happens in numpy whatever the backend: the vocabulary-sized
        # vector is small, the operations on it are sequential, and keeping
        # one code path means a seed reproduces the same text everywhere.
        logits = b.to_numpy(logits)

        if not np.isfinite(logits).all():
            raise ValueError(
                "The forward pass produced non-finite logits. The weights in "
                "this database do not form a working model."
            )
        return logits
