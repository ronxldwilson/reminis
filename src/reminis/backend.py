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

    def matmul_weight(self, x, w):
        """x @ w.T, whether w is a plain matrix or a packed one."""
        return x @ w.T

    # -- the pieces of a forward pass -------------------------------------

    def rms_norm(self, x, weight, eps: float):
        raise NotImplementedError

    def softmax(self, x, axis=-1):
        raise NotImplementedError

    def silu(self, x):
        raise NotImplementedError


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

    def pack(self, arr, bits: int, group_size: int = 32):
        mx = self.mx
        q, scales, biases = mx.quantize(arr, group_size=group_size, bits=bits)
        mx.eval(q, scales, biases)
        return QuantizedWeight(q, scales, biases, group_size, bits, arr.shape)

    def matmul_weight(self, x, w):
        if not isinstance(w, QuantizedWeight):
            return x @ w.T
        return self.mx.quantized_matmul(
            x, w.q, w.scales, w.biases, transpose=True,
            group_size=w.group_size, bits=w.bits,
        )

    def rms_norm(self, x, weight, eps: float):
        return self.mx.fast.rms_norm(x, weight, eps)

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
