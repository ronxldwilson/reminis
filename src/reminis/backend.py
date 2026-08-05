"""Pick the array library that suits the work and the hardware.

numpy is always there and is the reference implementation: everything else is
checked against it. Two others are used when they are installed and when the
work is the kind they are good at:

    mlx     Apple silicon. Unified memory, so weights are not copied to a
            device, and float16 is a native compute type rather than
            something numpy has to widen first.
    cupy    NVIDIA. Close enough to numpy's API to be nearly a drop-in.

The choice is per *workload*, not per machine, because a GPU is not a
blanket improvement. Every entry below was measured rather than assumed, and
one of them contradicted the assumption outright:

    inference   GPU wins, hugely on prompt processing -- 7-21x depending on
                the model -- where the work is one large matrix multiply.
    merge       GPU loses, 0.4-0.6x, and the road to that answer is the
                cautionary tale. A benchmark of one block said 6.6x faster;
                the real merge came out 2.4x slower. The block was warm and
                reused, while a merge reads each blob out of SQLite once and
                pays device-transfer costs the benchmark never showed.
    diff        GPU loses, and cannot win. On a 4M-element block the XOR is
                0.4 ms and the zlib compression is 252 ms, so it is entirely
                a compression workload and zlib runs on the CPU.

So `select("inference")` may hand back mlx while `select("elementwise")` hands
back numpy on the very same machine. Anything not measured defaults to numpy,
because an unproven GPU path is a slower path with more ways to be wrong --
and as the merge row shows, a benchmark of the wrong thing counts as unproven.
"""

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class QuantizedWeight:
    """A weight matrix kept packed, for a backend that multiplies it packed.

    GGML's K-quants cannot be multiplied directly by any of these libraries,
    so a quantized model is normally unpacked to float16 at load and the
    memory saving is lost. This keeps the saving by re-packing into the
    backend's own quantization format instead, at the cost of a second
    rounding on top of the one the file already has.
    """

    q: object
    scales: object
    biases: object
    group_size: int
    bits: int
    shape: tuple

# Workload names, and which backends are worth trying for each in order of
# preference. A name absent from here gets numpy.
PREFERENCE = {
    # A transformer forward pass: big matrix multiplies, weights reused
    # across many tokens, and float16 that a GPU can consume directly.
    "inference": ("mlx", "cupy", "numpy"),
    # Combining tensors elementwise during a merge. numpy, and the road to
    # that answer is why this table exists. A microbenchmark of one 4M-element
    # block said the GPU was 6.6x faster, because numpy's float16 decode and
    # encode run at ~1.2 GB/s and a GPU reads float16 natively. Wiring it into
    # the real merge made it 2.4x *slower*: a merge reads each blob out of
    # SQLite once, cold, and building a device array from a fresh buffer and
    # synchronising back costs more than the conversion saves. The block
    # benchmark reused one warm buffer and measured something else entirely.
    "elementwise": ("numpy",),
    # Byte-level work: XOR deltas, compression, hashing. Measured on a
    # 4M-element block, the XOR is 0.4 ms and zlib is 252 ms -- so this is
    # 100% a compression workload, and no GPU touches zlib.
    "bytes": ("numpy",),
}

# Workloads that must not lose precision relative to the numpy reference are
# run in float32 even on backends whose reason for existing is float16. A
# merge writes weights that are then stored forever; inference produces one
# token and is checked by argmax.
FLOAT32_WORKLOADS = frozenset({"elementwise"})

_ENV_OVERRIDE = "REMINIS_BACKEND"


class Backend:
    """The array operations reminis needs, over one array library."""

    name = "abstract"

    @classmethod
    def available(cls) -> bool:
        raise NotImplementedError

    @classmethod
    def describe(cls) -> str:
        return cls.name

    @classmethod
    def float32_dtype(cls):
        """This backend's float32, for workloads that must not lose precision."""
        return np.float32

    # -- moving data in and out ------------------------------------------

    def from_bytes(self, blob, dtype: str, shape=None):
        """Raw stored bytes to an array in this backend's compute dtype."""
        raise NotImplementedError

    def from_numpy(self, arr: np.ndarray):
        """A numpy array as one of this backend's, in the compute dtype."""
        raise NotImplementedError

    def to_numpy(self, x) -> np.ndarray:
        raise NotImplementedError

    def to_bytes(self, x, dtype: str) -> bytes:
        """An array back to raw stored bytes of the named dtype."""
        raise NotImplementedError

    def eval(self, *arrays):
        """Force evaluation, for backends that are lazy. A no-op otherwise."""

    @property
    def xp(self):
        """The array namespace, for the operations that need no adapting."""
        raise NotImplementedError

    def errstate(self):
        """Silence floating-point warnings, where the backend raises them."""
        import contextlib

        return contextlib.nullcontext()

    def zeros(self, shape):
        return self.xp.zeros(shape, dtype=self.compute_dtype)

    def contiguous(self, x):
        return x

    # -- packed weights ----------------------------------------------------

    def can_pack(self) -> bool:
        """Whether this backend can multiply by a weight without unpacking it."""
        return False

    def pack(self, arr, bits: int, group_size: int = 32):
        raise NotImplementedError

    def adopt_packed(self, words, scales, biases, bits, group_size, shape):
        """Take an already-packed weight, laid out the way this backend wants."""
        raise NotImplementedError

    def matmul_weight(self, x, w):
        """x @ w.T, whether w is a plain matrix or a packed one."""
        return x @ w.T

    def take_rows(self, w, indices):
        """Rows of a weight matrix, unpacking only the ones asked for.

        An embedding table is indexed rather than multiplied, which is why
        it is normally left unpacked -- but a packed table can still be
        indexed, because the rows are stored contiguously. Only the handful
        of rows a prompt actually needs get decoded.
        """
        return self.xp.take(w, indices, axis=0)

    def gather_matmul(self, x, w, indices, k):
        """Multiply each token by the experts chosen for it.

        `x` is (tokens, features), `indices` is (tokens, k) naming a row of
        the stacked expert tensor, and the result is (tokens, k, out). The
        fallback gathers the chosen experts into one batch and does a single
        batched multiply, so nothing loops over experts in Python.
        """
        xp = self.xp
        n_tokens = x.shape[0]
        gathered = xp.take(w, indices.reshape(-1), axis=0)
        rows = xp.repeat(x, k, axis=0)[:, None, :]
        out = rows @ xp.swapaxes(gathered, -1, -2)
        return out.reshape(n_tokens, k, -1)

    # -- the pieces of a forward pass -------------------------------------

    def rms_norm(self, x, weight, eps: float):
        raise NotImplementedError

    def softmax(self, x, axis=-1):
        raise NotImplementedError

    def silu(self, x):
        raise NotImplementedError

    def to_compute32(self, x):
        """Widen to float32, for a reduction half precision would spoil.

        Routing is the case that matters: a softmax over experts decides
        which ones run at all, so a rounding there changes the computation
        rather than nudging it.
        """
        return x

    def rope(self, x, dims, traditional, base, offset, freqs=None):
        """Rotary embedding on (batch, heads, tokens, dim).

        Backends with a fused kernel for this override it; the fallback
        builds the cos/sin tables and does the rotation as array ops.
        """
        xp = self.xp
        half = dims // 2
        # `freqs`, when given, is the *period* of each rotary dimension --
        # base ** (2i/d) -- which is the convention mlx's kernel uses, so the
        # angle divides by it rather than multiplying.
        if freqs is None:
            periods = base ** (np.arange(half, dtype=np.float32) * 2.0 / dims)
        else:
            periods = np.asarray(freqs, dtype=np.float32)
        pos = np.arange(offset, offset + x.shape[-2], dtype=np.float32)
        angles = pos[:, None] / periods[None, :]
        cos = self.from_numpy(np.cos(angles))[None, None]
        sin = self.from_numpy(np.sin(angles))[None, None]

        rot, rest = x[..., :dims], x[..., dims:]
        if traditional:
            even, odd = rot[..., 0::2], rot[..., 1::2]
            out = xp.stack([even * cos - odd * sin, even * sin + odd * cos],
                           axis=-1).reshape(rot.shape)
        else:
            first, second = rot[..., :half], rot[..., half:]
            out = xp.concatenate(
                [first * cos - second * sin, first * sin + second * cos], axis=-1
            )
        return out if rest.shape[-1] == 0 else xp.concatenate([out, rest], axis=-1)

    def fused_ffn(self, x, norm, gate_up, down, eps):
        """norm, gated feed-forward, and the residual add, as one step.

        Written as a single call so that a backend able to fuse it can,
        rather than seeing seven unrelated operations arrive one at a time.
        """
        h = self.rms_norm(x, norm, eps)
        out = self.matmul_weight(h, gate_up)
        half = out.shape[-1] // 2
        return x + self.matmul_weight(
            self.silu(out[..., :half]) * out[..., half:], down
        )

    def attention(self, q, k, v, scale, mask=None, sinks=None):
        """Scaled dot-product attention on (batch, heads, tokens, dim).

        Grouped-query attention is expressed by k and v having fewer heads
        than q. The fallback below spells that out with a broadcast axis
        rather than repeating the cache, which would copy it every layer.

        `sinks` is one learned logit per head that joins the softmax
        denominator without contributing a value -- an escape valve letting
        a head attend to nothing in particular. Implemented by carrying the
        sink as an extra score column and dropping it after the softmax,
        which is what the fused kernels do internally.
        """
        xp = self.xp
        n_heads, n_kv = q.shape[1], k.shape[1]
        repeat = n_heads // n_kv
        b, _, t, d = q.shape

        qh = q.reshape(b, n_kv, repeat, t, d)
        kh = k[:, :, None]
        vh = v[:, :, None]

        scores = (qh @ xp.swapaxes(kh, -1, -2)) * scale
        if mask is not None:
            scores = xp.where(mask, scores, float("-inf"))

        if sinks is None:
            attn = self.softmax(scores)
        else:
            column = xp.broadcast_to(
                sinks.reshape(n_kv, repeat, 1, 1).astype(scores.dtype),
                scores.shape[:-1] + (1,),
            )
            attn = self.softmax(xp.concatenate([scores, column], axis=-1))
            attn = attn[..., :-1]

        return (attn @ vh).reshape(b, n_heads, t, d)


class NumpyBackend(Backend):
    """The reference implementation. Always available, always correct."""

    name = "numpy"
    # numpy has no BLAS-backed float16 matmul -- BLAS provides sgemm and
    # dgemm and no half-precision equivalent -- so float16 falls back to a
    # generic loop that measured 53x slower than float32 on the same
    # product. Everything is therefore widened once, at load.
    compute_dtype = np.float32

    @classmethod
    def available(cls) -> bool:
        return True

    @classmethod
    def describe(cls) -> str:
        return f"numpy {np.__version__} (CPU)"

    def from_bytes(self, blob, dtype: str, shape=None):
        from reminis.dtypes import to_float32

        arr = to_float32(blob, dtype)
        return arr.reshape(shape) if shape is not None else arr

    def from_numpy(self, arr: np.ndarray):
        return np.asarray(arr, dtype=np.float32)

    def to_numpy(self, x) -> np.ndarray:
        return x

    def to_bytes(self, x, dtype: str) -> bytes:
        from reminis.dtypes import from_float32

        return from_float32(x, dtype)

    @property
    def xp(self):
        return np

    def errstate(self):
        # Apple's Accelerate raises the divide-by-zero and overflow flags
        # during ordinary float32 matmuls whose inputs and outputs are all
        # finite, and the causal mask is built from -inf on purpose. Neither
        # is a problem, and both would otherwise warn on every token.
        return np.errstate(all="ignore")

    def contiguous(self, x):
        return np.ascontiguousarray(x)

    def rms_norm(self, x, weight, eps: float):
        var = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True)
        return (x * np.reciprocal(np.sqrt(var + eps))) * weight

    def softmax(self, x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        np.exp(x, out=x)
        return x / np.sum(x, axis=axis, keepdims=True)

    def silu(self, x):
        return x / (1.0 + np.exp(-x, dtype=np.float32))


class MLXBackend(Backend):
    """Apple silicon, through mlx.

    Weights stay in float16 rather than being widened, which halves both the
    memory they occupy and the bandwidth reading them costs -- and on this
    hardware that is most of what token generation is doing. Normalisation
    and softmax are still computed in float32, since those reduce over a
    whole row and are where half precision actually hurts.

    Unified memory is the reason this is worth doing at all: there is no
    host-to-device copy to amortise, which is usually what eats the gain on
    a model small enough to be interesting.
    """

    name = "mlx"

    def __init__(self, compute_dtype=None):
        import mlx.core as mx

        self.mx = mx
        self.compute_dtype = compute_dtype or mx.float16
        self._compiled_ffn = None

    @classmethod
    def available(cls) -> bool:
        try:
            import mlx.core as mx
        except ImportError:
            return False
        try:
            # A metal device is the whole point; mlx on a machine without
            # one would be a slower numpy.
            return mx.metal.is_available()
        except AttributeError:
            try:
                return mx.default_device().type == mx.DeviceType.gpu
            except Exception:
                return True

    @classmethod
    def float32_dtype(cls):
        import mlx.core as mx

        return mx.float32

    @classmethod
    def describe(cls) -> str:
        try:
            import mlx.core as mx
            return f"mlx {mx.__version__} ({mx.default_device()})"
        except ImportError:
            return "mlx (unavailable)"

    def from_bytes(self, blob, dtype: str, shape=None):
        mx = self.mx
        # bfloat16 is the case numpy cannot represent at all, and mlx can,
        # so its bytes are reinterpreted rather than widened.
        if dtype == "BF16":
            arr = mx.array(np.frombuffer(blob, dtype=np.uint16)).view(mx.bfloat16)
        elif dtype == "F16":
            arr = mx.array(np.frombuffer(blob, dtype=np.float16))
        elif dtype == "F32":
            arr = mx.array(np.frombuffer(blob, dtype=np.float32))
        elif dtype == "F64":
            arr = mx.array(np.frombuffer(blob, dtype=np.float64).astype(np.float32))
        else:
            raise ValueError(f"Cannot decode dtype '{dtype}'")
        arr = arr.astype(self.compute_dtype)
        return arr.reshape(shape) if shape is not None else arr

    def from_numpy(self, arr: np.ndarray):
        return self.mx.array(np.ascontiguousarray(arr, dtype=np.float32)).astype(
            self.compute_dtype
        )

    def to_numpy(self, x) -> np.ndarray:
        return np.array(x.astype(self.mx.float32))

    def to_bytes(self, x, dtype: str) -> bytes:
        mx = self.mx
        # Narrowing happens on the device, which is the point: numpy encodes
        # float16 at about 1.2 GB/s and this does not have to.
        if dtype == "F16":
            return np.array(x.astype(mx.float16)).tobytes()
        if dtype == "F32":
            return np.array(x.astype(mx.float32)).tobytes()
        if dtype == "BF16":
            # mlx rounds to nearest even on this cast, matching what
            # reminis.dtypes does by hand for numpy.
            return np.array(x.astype(mx.bfloat16).view(mx.uint16)).tobytes()
        from reminis.dtypes import from_float32

        return from_float32(self.to_numpy(x), dtype)

    def eval(self, *arrays):
        self.mx.eval(*arrays)

    @property
    def xp(self):
        return self.mx

    def can_pack(self) -> bool:
        return True

    def pack(self, arr, bits: int, group_size: int = 32, compact=False):
        mx = self.mx
        q, scales, biases = mx.quantize(arr, group_size=group_size, bits=bits)
        if compact:
            scales, biases = scales.astype(mx.float16), biases.astype(mx.float16)
        mx.eval(q, scales, biases)
        return QuantizedWeight(q, scales, biases, group_size, bits, arr.shape)

    def adopt_packed(self, words, scales, biases, bits, group_size, shape,
                     compact=False):
        """Take an already-packed weight in this backend's layout.

        `compact` halves the per-group overhead by holding the scale and
        bias in float16 rather than float32. At a group size of 32 that is
        the difference between 6 and 5 bits per weight for a 4-bit tensor --
        17% of the model -- and it costs a measured 8.3e-04 relative error,
        which is far below the quantization already in the file. It is the
        one thing that makes the repack stop being bit-exact, so it is opt-in.
        """
        mx = self.mx
        dtype = mx.float16 if compact else mx.float32
        q = mx.array(words)
        s = mx.array(scales).astype(dtype)
        b = mx.array(biases).astype(dtype)
        mx.eval(q, s, b)
        return QuantizedWeight(q, s, b, group_size, bits, shape)

    def matmul_weight(self, x, w):
        if not isinstance(w, QuantizedWeight):
            return x @ w.T
        return self.mx.quantized_matmul(
            x, w.q, w.scales, w.biases, transpose=True,
            group_size=w.group_size, bits=w.bits,
        )

    def to_compute32(self, x):
        return x.astype(self.mx.float32)

    def take_rows(self, w, indices):
        if not isinstance(w, QuantizedWeight):
            return self.mx.take(w, indices, axis=0)
        mx = self.mx
        rows = mx.take(w.q, indices, axis=0)
        scales = mx.take(w.scales, indices, axis=0)
        biases = mx.take(w.biases, indices, axis=0)
        return mx.dequantize(rows, scales, biases,
                             group_size=w.group_size, bits=w.bits)

    def gather_matmul(self, x, w, indices, k):
        if not isinstance(w, QuantizedWeight):
            return Backend.gather_matmul(self, x, w, indices, k)
        mx = self.mx
        n_tokens, features = x.shape
        out = mx.gather_qmm(
            x.reshape(n_tokens, 1, 1, features), w.q, w.scales, w.biases,
            rhs_indices=indices.astype(mx.uint32), transpose=True,
            group_size=w.group_size, bits=w.bits,
        )
        return out.reshape(n_tokens, k, -1)

    def rms_norm(self, x, weight, eps: float):
        return self.mx.fast.rms_norm(x, weight, eps)

    def rope(self, x, dims, traditional, base, offset, freqs=None):
        # A fused kernel, which matters because the array-op version below
        # builds half a dozen intermediates per layer per token.
        return self.mx.fast.rope(
            x, dims, traditional=traditional,
            base=None if freqs is not None else base,
            scale=1.0, offset=offset,
            freqs=None if freqs is None else self.mx.array(np.asarray(freqs, np.float32)),
        )

    def fused_ffn(self, x, norm, gate_up, down, eps):
        """The feed-forward half of a block, compiled into one graph.

        Every layer of a model has the same shapes here, so one compiled
        function serves all of them -- the weights are traced arguments
        rather than constants. Compiling removes the per-operation dispatch
        that a small model spends a quarter of its time in.

        Packed weights are not traceable arguments, so those fall back to
        the uncompiled path.
        """
        if isinstance(gate_up, QuantizedWeight) or isinstance(down, QuantizedWeight):
            return Backend.fused_ffn(self, x, norm, gate_up, down, eps)

        mx = self.mx
        if self._compiled_ffn is None:
            def ffn(x, norm, gate_up, down):
                h = mx.fast.rms_norm(x, norm, eps)
                out = x
                proj = h @ gate_up.T
                half = proj.shape[-1] // 2
                gate, up = proj[..., :half], proj[..., half:]
                return out + (gate * mx.sigmoid(gate) * up) @ down.T

            self._compiled_ffn = mx.compile(ffn)
        return self._compiled_ffn(x, norm, gate_up, down)

    def attention(self, q, k, v, scale, mask=None, sinks=None):
        # Fused attention: the scores matrix is never materialised, and
        # grouped-query attention and sinks are handled inside the kernel.
        return self.mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask, sinks=sinks
        )

    def softmax(self, x, axis=-1):
        # precise=True accumulates in float32 even when the input is half,
        # which is exactly the reduction where half precision goes wrong.
        return self.mx.softmax(x, axis=axis, precise=True)

    def silu(self, x):
        import mlx.nn as nn

        return nn.silu(x)


class CupyBackend(Backend):
    """NVIDIA, through cupy.

    cupy mirrors numpy's API closely enough that the forward pass needs no
    special cases beyond getting arrays onto the device and back. Unlike
    mlx there *is* a host-to-device copy, but it happens once per weight at
    load rather than per token, so it amortises over any real generation.

    This path is written but unverified: there is no NVIDIA hardware here to
    run it on. It is selected only when cupy imports and reports a device,
    and `reminis run --backend numpy` is always there if it misbehaves.
    """

    name = "cupy"

    def __init__(self, compute_dtype=None):
        import cupy as cp

        self.cp = cp
        self.compute_dtype = compute_dtype or cp.float16

    @classmethod
    def available(cls) -> bool:
        try:
            import cupy as cp
        except ImportError:
            return False
        try:
            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False

    @classmethod
    def float32_dtype(cls):
        import cupy as cp

        return cp.float32

    @classmethod
    def describe(cls) -> str:
        try:
            import cupy as cp
            name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            return f"cupy {cp.__version__} ({name})"
        except Exception:
            return "cupy (unavailable)"

    def from_bytes(self, blob, dtype: str, shape=None):
        from reminis.dtypes import to_float32

        cp = self.cp
        if dtype == "F16":
            host = np.frombuffer(blob, dtype=np.float16)
        elif dtype == "F32":
            host = np.frombuffer(blob, dtype=np.float32)
        else:
            # BF16 and F64 go through numpy's widening; cupy has no
            # bfloat16, so there is nothing to be gained by being clever.
            host = to_float32(blob, dtype)
        arr = cp.asarray(host).astype(self.compute_dtype)
        return arr.reshape(shape) if shape is not None else arr

    def from_numpy(self, arr: np.ndarray):
        return self.cp.asarray(np.ascontiguousarray(arr, dtype=np.float32)).astype(
            self.compute_dtype
        )

    def to_numpy(self, x) -> np.ndarray:
        return self.cp.asnumpy(x.astype(self.cp.float32))

    def to_bytes(self, x, dtype: str) -> bytes:
        cp = self.cp
        if dtype == "F16":
            return cp.asnumpy(x.astype(cp.float16)).tobytes()
        if dtype == "F32":
            return cp.asnumpy(x.astype(cp.float32)).tobytes()
        from reminis.dtypes import from_float32

        return from_float32(self.to_numpy(x), dtype)

    def eval(self, *arrays):
        self.cp.cuda.Stream.null.synchronize()

    @property
    def xp(self):
        return self.cp

    def contiguous(self, x):
        return self.cp.ascontiguousarray(x)

    def rms_norm(self, x, weight, eps: float):
        cp = self.cp
        x32 = x.astype(cp.float32)
        var = cp.mean(cp.square(x32), axis=-1, keepdims=True)
        out = x32 * cp.reciprocal(cp.sqrt(var + eps))
        return (out * weight.astype(cp.float32)).astype(x.dtype)

    def softmax(self, x, axis=-1):
        cp = self.cp
        x32 = x.astype(cp.float32)
        x32 = x32 - cp.max(x32, axis=axis, keepdims=True)
        cp.exp(x32, out=x32)
        return (x32 / cp.sum(x32, axis=axis, keepdims=True)).astype(x.dtype)

    def silu(self, x):
        cp = self.cp
        return x * (1.0 / (1.0 + cp.exp(-x.astype(cp.float32)))).astype(x.dtype)


_BACKENDS = {"numpy": NumpyBackend, "mlx": MLXBackend, "cupy": CupyBackend}


def available_backends() -> list[str]:
    """Which backends this machine could actually use, best first."""
    return [n for n in ("mlx", "cupy", "numpy") if _BACKENDS[n].available()]


def select(workload: str = "inference", requested: str | None = None) -> Backend:
    """The backend to use for a kind of work.

    An explicit request wins and is an error if it cannot be honoured --
    silently falling back would make a benchmark comparing backends compare
    numpy with numpy. Otherwise the preference list for the workload is
    walked until something is available.
    """
    def build(cls):
        if workload in FLOAT32_WORKLOADS and cls is not NumpyBackend:
            return cls(compute_dtype=cls.float32_dtype())
        return cls()

    requested = requested or os.environ.get(_ENV_OVERRIDE)
    if requested and requested != "auto":
        cls = _BACKENDS.get(requested)
        if cls is None:
            raise ValueError(
                f"Unknown backend '{requested}'; expected one of "
                f"{', '.join(_BACKENDS)}"
            )
        if not cls.available():
            raise ValueError(
                f"Backend '{requested}' is not available here. "
                f"This machine has: {', '.join(available_backends())}"
            )
        return build(cls)

    for name in PREFERENCE.get(workload, ("numpy",)):
        cls = _BACKENDS[name]
        if cls.available():
            return build(cls)
    return NumpyBackend()


def report() -> str:
    """A line per backend, for `reminis info` and bug reports."""
    lines = []
    for name in ("numpy", "mlx", "cupy"):
        cls = _BACKENDS[name]
        mark = "yes" if cls.available() else "no "
        lines.append(f"  {mark}  {cls.describe()}")
    lines.append("")
    for workload in ("inference", "elementwise", "bytes"):
        lines.append(f"  {workload:12s} -> {select(workload).name}")
    return "\n".join(lines)
