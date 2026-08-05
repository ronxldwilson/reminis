"""Dtype tables and float conversions shared across formats.

reminis stores every tensor as raw bytes plus a dtype *name*. Two dtype systems
can appear in a database: GGML types (from GGUF) and safetensors types. The
names overlap -- both call 32-bit floats ``F32`` -- but the numeric ids behind
them do not, so ``reminis.dtype_system`` in ``model_meta`` records which system
a database's ``dtype_id`` column belongs to. Nothing should interpret
``dtype_id`` without checking it.

BF16 is the reason this module exists. It is the dtype most modern fine-tuning
produces, numpy has no native type for it, and reading its bytes as float16
yields silent garbage rather than an error. Every numeric path goes through
``to_float32`` / ``from_float32`` so that stays impossible.
"""

import numpy as np

DTYPE_SYSTEM_GGUF = "gguf"
DTYPE_SYSTEM_SAFETENSORS = "safetensors"

# Safetensors dtype names and their sizes in bytes. Ids are reminis-local: the
# format itself identifies dtypes by name only, so we assign stable integers
# purely to fill the `dtype_id` column.
SAFETENSORS_DTYPES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

SAFETENSORS_DTYPE_IDS = {name: i for i, name in enumerate(SAFETENSORS_DTYPES)}

# Dtypes that map one-to-one onto a numpy type of the same width.
_DIRECT_FLOATS = {"F64": np.float64, "F32": np.float32, "F16": np.float16}

# GGML type names that mean the same thing as the safetensors name. Quantized
# GGML types (Q4_K, IQ3_M, ...) have no safetensors equivalent and are absent
# here on purpose -- exporting them must fail loudly, not guess.
GGML_TO_SAFETENSORS_DTYPE = {
    "F64": "F64",
    "F32": "F32",
    "F16": "F16",
    "BF16": "BF16",
    "I8": "I8",
    "I16": "I16",
    "I32": "I32",
    "I64": "I64",
}


def is_float_dtype(dtype: str) -> bool:
    """True when the dtype's bytes decode to real numbers we can do math on."""
    return dtype in _DIRECT_FLOATS or dtype == "BF16"


def to_float32(blob: bytes, dtype: str) -> np.ndarray:
    """Decode raw tensor bytes to a float32 array.

    Always returns a fresh, writable array -- callers reshape and mutate it.
    """
    if dtype == "BF16":
        # BF16 is exactly the top 16 bits of an F32, so widening is a shift.
        return (
            np.frombuffer(blob, dtype=np.uint16).astype(np.uint32) << 16
        ).view(np.float32)

    np_dtype = _DIRECT_FLOATS.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Cannot decode dtype '{dtype}' to floats")
    return np.frombuffer(blob, dtype=np_dtype).astype(np.float32)


def from_float32(arr: np.ndarray, dtype: str) -> bytes:
    """Encode a float32 array back into raw bytes of the given dtype."""
    if dtype == "BF16":
        return _f32_to_bf16(arr).tobytes()

    np_dtype = _DIRECT_FLOATS.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Cannot encode floats as dtype '{dtype}'")
    return np.ascontiguousarray(arr).astype(np_dtype).tobytes()


def _f32_to_bf16(arr: np.ndarray) -> np.ndarray:
    """Narrow float32 to bfloat16 with round-to-nearest-even.

    Plain truncation would be simpler but biases every value toward zero, and
    that bias accumulates over a stack of tensors. Round-to-nearest-even is
    what PyTorch does, so a reminis round-trip agrees with a torch one.

    The uint32 arithmetic below wraps rather than saturating, which is fine for
    every finite value (the largest, 0xFF7FFFFF, cannot overflow by 0x8000) but
    would mangle NaN payloads, so NaN is written explicitly afterwards.
    """
    x = np.ascontiguousarray(arr, dtype=np.float32)
    u = x.view(np.uint32)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(x)
    if nan.any():
        rounded = rounded.copy()
        rounded[nan] = 0x7FC0
    return rounded


def dtype_system(conn) -> str:
    """Which dtype system a model database's `dtype_id` column refers to.

    Databases written before safetensors support carry no marker and are
    always GGUF, so that is the default.
    """
    row = conn.execute(
        "SELECT value FROM model_meta WHERE key = 'reminis.dtype_system'"
    ).fetchone()
    return row[0] if row else DTYPE_SYSTEM_GGUF
