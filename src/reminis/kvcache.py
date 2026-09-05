"""Keys and values, kept across steps so a token is not recomputed.

The cache is the part of inference that grows with the conversation where
the weights do not, so on a long context it is what decides whether the
model fits at all. Two things follow from that and are both here: the
buffers double rather than reallocate per token, and ``--kv-bits`` stores
them quantized, which costs speed rather than saving it -- there is no
quantized attention kernel, so the cache is decompressed to attend.
"""

from reminis.backend import select as select_backend
from reminis.errors import UnsupportedModel

class KVCache:
    """Keys and values for every layer.

    Growing this with `np.concatenate` reallocates and recopies the entire
    cache on every token, which turns a linear cost into a quadratic one and
    is invisible until the context is long. Given a capacity up front it
    allocates once and writes each token into place, returning a view.
    Without one it still grows, in doubling steps rather than by one.
    """

    def __init__(self, n_layers: int, capacity: int | None = None, backend=None,
                 quantize_bits: int | None = None,
                 last_append_layer: int | None = None):
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.capacity = capacity
        self.backend = backend or select_backend("inference")
        # Compressing the cache trades arithmetic for room. The weights are
        # a fixed cost, but the cache grows with every token, so at long
        # context it is the cache that decides whether a prompt fits at all.
        self.quantize_bits = quantize_bits
        self._packed_k = [None] * n_layers
        self._packed_v = [None] * n_layers
        self._used = 0
        self._last_append = last_append_layer if last_append_layer is not None else n_layers - 1

    def _empty(self, size, like, n_tokens):
        """A cache buffer shaped like `like` but with room for `size` tokens."""
        xp = self.backend.xp
        shape = list(like.shape)
        shape[-2] = size
        return xp.zeros(tuple(shape), dtype=like.dtype)

    def append(self, layer: int, k, v):
        if self.quantize_bits:
            return self._append_quantized(layer, k, v)
        n = k.shape[-2]
        buf_k, buf_v = self.k[layer], self.v[layer]

        if buf_k is None:
            size = max(self.capacity or 0, n)
            buf_k = self._empty(size, k, n)
            buf_v = self._empty(size, v, n)
            self.k[layer], self.v[layer] = buf_k, buf_v
            used = 0
        else:
            used = self._used
            if used + n > buf_k.shape[-2]:
                grown = max(buf_k.shape[-2] * 2, used + n)
                bigger_k = self._empty(grown, k, n)
                bigger_v = self._empty(grown, v, n)
                bigger_k[..., :used, :] = buf_k[..., :used, :]
                bigger_v[..., :used, :] = buf_v[..., :used, :]
                buf_k, buf_v = bigger_k, bigger_v
                self.k[layer], self.v[layer] = buf_k, buf_v

        buf_k[..., used:used + n, :] = k
        buf_v[..., used:used + n, :] = v

        # The counter advances once per token, not once per layer, so it is
        # updated on the last layer only -- every layer sees the same span.
        if layer == self._last_append:
            self._used = used + n
        return buf_k[..., :used + n, :], buf_v[..., :used + n, :]

    def _append_quantized(self, layer: int, k, v):
        """Keep the cache compressed, decompressing it to attend.

        There is no quantized attention kernel to hand, so this saves room
        rather than time: the span is decompressed on every layer of every
        token. It is worth it when the cache is what stops a prompt fitting,
        and not otherwise -- which is why it is off unless asked for.
        """
        b = self.backend
        group = 64 if k.shape[-1] % 64 == 0 else 32
        n = k.shape[-2]
        used = self._used

        for store, value in ((self._packed_k, k), (self._packed_v, v)):
            packed = b.quantize_kv(value, self.quantize_bits, group)
            if packed is None:
                raise UnsupportedModel(
                    f"The {b.name} backend cannot compress a KV cache."
                )
            buffers = store[layer]
            if buffers is None:
                # Preallocated for the same reason the uncompressed cache is:
                # growing by concatenation recopies everything every token,
                # which is quadratic in a context length that is the whole
                # point of compressing it.
                size = max(self.capacity or 0, used + n)
                buffers = tuple(
                    self._sized_like(part, size) for part in packed
                )
                store[layer] = buffers
            elif used + n > buffers[0].shape[-2]:
                grown = max(buffers[0].shape[-2] * 2, used + n)
                bigger = tuple(self._sized_like(part, grown) for part in packed)
                for old, new in zip(buffers, bigger):
                    new[..., :used, :] = old[..., :used, :]
                buffers = bigger
                store[layer] = buffers

            for buffer, part in zip(buffers, packed):
                buffer[..., used:used + n, :] = part

        if layer == self._last_append:
            self._used = used + n

        span = used + n
        return (
            b.dequantize_kv(tuple(x[..., :span, :] for x in self._packed_k[layer])
                            + (self.quantize_bits, group)),
            b.dequantize_kv(tuple(x[..., :span, :] for x in self._packed_v[layer])
                            + (self.quantize_bits, group)),
        )

    def _sized_like(self, part, size):
        shape = list(part.shape)
        shape[-2] = size
        return self.backend.xp.zeros(tuple(shape), dtype=part.dtype)

    @property
    def length(self) -> int:
        return self._used

    def rollback(self, length: int) -> None:
        """Forget every token past `length`, keeping the ones before it.

        Speculative decoding runs tokens through the model before it knows
        whether they are wanted, so it needs a way to un-run the ones that
        turn out not to be. Nothing has to be erased: the buffers are
        preallocated and every read is bounded by the counter, so lowering
        the counter is what makes the rejected span invisible, and the next
        token overwrites it. That is the whole of the rollback the
        attention layers need, and it costs nothing.

        What it does *not* cover is a layer that carries state rather than
        keys and values. A recurrent layer folds each token into a hidden
        state and keeps no per-token record to truncate, so rolling one
        back is `Model.snapshot_state`, not this.
        """
        if not 0 <= length <= self._used:
            raise ValueError(
                f"Cannot roll a cache holding {self._used} tokens back to "
                f"{length}"
            )
        self._used = length
