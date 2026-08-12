"""Quantize a model database down to Q8_0 or Q4_0.

reminis could unpack every GGML quantization and never produce one, so a float
model stayed a float model. This writes the two simplest GGML formats, chosen
because they are exactly specified, cheap to compute, and -- being real GGUF
types -- keep the result portable: it exports back to GGUF, llama.cpp reads it,
and every reminis backend runs it. A backend-specific packing would have been
faster to write and would have produced a file only this machine understood.

  Q8_0   34 bytes per 32 weights   w = d * q          d is float16, q is int8
  Q4_0   18 bytes per 32 weights   w = d * (q - 8)    q is a 4-bit nibble

Both are round-to-nearest against a per-block scale taken from the block's
largest magnitude, which is what llama.cpp's own quantizer does. Neither uses
an importance matrix, so this is not trying to beat `llama-quantize` on
quality -- it is trying to be the fast, obvious thing that works.

Only 2-D tensors whose rows are a multiple of 32 are quantized. Norms, biases
and anything 1-D stay exactly as they are, which is what GGUF does too: they
are a rounding error of the file and the first thing to damage a model.
"""

import ast
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from gguf.constants import GGMLQuantizationType

from reminis.db import INSERT_TENSOR, open_read_only
from reminis.dtypes import is_float_dtype, to_float32

BLOCK = 32

# name -> (block bytes, GGML type id)
FORMATS = {
    "Q8_0": (34, int(GGMLQuantizationType.Q8_0)),
    "Q4_0": (18, int(GGMLQuantizationType.Q4_0)),
}

BITS_TO_FORMAT = {8: "Q8_0", 4: "Q4_0"}


def quantize_q8_0(arr: np.ndarray) -> bytes:
    """Rows of float32 to Q8_0 blocks.

    The scale is the block's largest magnitude over 127, so the extreme value
    lands on the edge of the range and everything else rounds inside it.
    """
    # asarray rather than astype: the caller already hands us float32, and
    # astype would copy the whole tensor to say so.
    blocks = np.asarray(arr, dtype=np.float32).reshape(-1, BLOCK)
    # max(|x|) without materialising |x|, which on a multi-hundred-megabyte
    # tensor is a temporary the size of the tensor. Two reductions instead:
    # the largest magnitude is either the biggest value or the most negative.
    amax = np.maximum(blocks.max(axis=1), -blocks.min(axis=1))
    d = (amax / 127.0).astype(np.float16)
    # Dividing by a zero scale would produce NaN for an all-zero block, which
    # is a real case in a trained model, not a pathological one.
    scale = d.astype(np.float32)
    inv = np.where(scale > 0, 1.0 / np.where(scale > 0, scale, 1.0), 0.0)
    # One scaled copy, then rounded and clipped in place, rather than a fresh
    # array per step.
    scaled = blocks * inv[:, None]
    np.rint(scaled, out=scaled)
    np.clip(scaled, -127, 127, out=scaled)
    q = scaled.astype(np.int8)

    out = np.empty((len(blocks), 34), dtype=np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def quantize_q4_0(arr: np.ndarray) -> bytes:
    """Rows of float32 to Q4_0 blocks.

    The scale is taken from the signed value with the largest magnitude,
    divided by -8, which is what puts that value at code 0 and centres the
    rest around 8. Taking `max(|x|)` unsigned instead would flip the sign of
    every weight in a block whose extreme is negative.
    """
    blocks = np.asarray(arr, dtype=np.float32).reshape(-1, BLOCK)
    # argmax over |x| rather than a max/min comparison, because the two
    # disagree when a block's largest positive and largest negative values
    # have the same magnitude: argmax takes whichever comes first, and that
    # is the encoding every earlier version of this wrote.
    idx = np.abs(blocks).argmax(axis=1)
    extreme = blocks[np.arange(len(blocks)), idx]
    d = (extreme / -8.0).astype(np.float16)

    scale = d.astype(np.float32)
    inv = np.where(scale != 0, 1.0 / np.where(scale != 0, scale, 1.0), 0.0)
    scaled = blocks * inv[:, None]
    np.rint(scaled, out=scaled)
    scaled += 8
    np.clip(scaled, 0, 15, out=scaled)
    q = scaled.astype(np.uint8)

    # GGML puts the first 16 weights in the low nibbles and the next 16 in the
    # high nibbles of the same bytes, rather than two consecutive weights per
    # byte. Getting this backwards produces a file that loads and is wrong.
    low, high = q[:, :16], q[:, 16:]
    out = np.empty((len(blocks), 18), dtype=np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = low | (high << 4)
    return out.tobytes()


QUANTIZERS = {"Q8_0": quantize_q8_0, "Q4_0": quantize_q4_0}

# How many 32-weight blocks one worker takes at a time. At 64k blocks a
# chunk's scaled copy is 8 MB, so the peak working set is a few tens of
# megabytes however large the tensor is -- an embedding matrix that would
# otherwise want a gigabyte of float32 all at once is handled in slices.
CHUNK_BLOCKS = 64 * 1024

# Above this, a tensor is worth handing to several threads. Below it the
# pool costs more than it saves.
PARALLEL_MIN_BLOCKS = CHUNK_BLOCKS * 2


def _quantize_chunked(arr: np.ndarray, quantizer, pool) -> bytes:
    """Quantize one tensor across several threads.

    Every block is scaled against its own extreme and nothing crosses a
    block boundary, so slicing the tensor into whole-block chunks and
    concatenating what comes back is exactly what one pass would produce.
    numpy drops the interpreter lock for the arithmetic, so the threads
    genuinely overlap -- measured 4-5x on a 1 GB tensor.
    """
    n_blocks = arr.size // BLOCK
    if pool is None or n_blocks < PARALLEL_MIN_BLOCKS:
        return quantizer(arr)

    def piece(start):
        stop = min(start + CHUNK_BLOCKS, n_blocks)
        return quantizer(arr[start * BLOCK:stop * BLOCK])

    return b"".join(pool.map(piece, range(0, n_blocks, CHUNK_BLOCKS)))


def _eligible(shape: list, dtype: str) -> bool:
    """Whether a tensor is worth quantizing and can be.

    Two dimensions, a row length that divides into whole blocks, and float
    input. A model that is already quantized has nothing to do here -- going
    through float32 to reach another quantization would round twice.
    """
    return (
        is_float_dtype(dtype)
        and len(shape) == 2
        and shape[0] % BLOCK == 0
    )


def quantize_model(
    db_path: str,
    output_path: str,
    bits: int = 4,
    verbose: bool = True,
) -> dict:
    """Write a quantized copy of a model database.

    Metadata and ineligible tensors are copied through unchanged, so the
    result is an ordinary reminis database that `run`, `info` and `export`
    all handle without knowing how it was made.
    """
    fmt = BITS_TO_FORMAT.get(bits)
    if fmt is None:
        raise ValueError(
            f"--bits takes 8 (Q8_0) or 4 (Q4_0); got {bits}. Widths between "
            "the two need a k-quant, which this does not write."
        )
    block_bytes, type_id = FORMATS[fmt]
    quantizer = QUANTIZERS[fmt]

    source = Path(db_path)
    if not source.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if Path(output_path).resolve() == source.resolve():
        raise ValueError("Refusing to quantize a database over itself; "
                         "pass a different -o/--output.")

    src = open_read_only(db_path)
    Path(output_path).unlink(missing_ok=True)
    from reminis.converter import SCHEMA, open_for_bulk_write
    out = open_for_bulk_write(output_path)
    out.executescript(SCHEMA)
    out.execute("BEGIN")

    out.executemany(
        "INSERT OR REPLACE INTO model_meta (key, value, type) VALUES (?,?,?)",
        src.execute("SELECT key, value, type FROM model_meta"),
    )

    started = time.perf_counter()
    stats = {"quantized": 0, "copied": 0, "raw_bytes": 0, "new_bytes": 0}

    rows = src.execute(
        "SELECT name, shape, dtype, dtype_id, n_elements, n_bytes, data "
        "FROM tensors ORDER BY id"
    )
    pool = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8))
    try:
        for name, shape_json, dtype, dtype_id, n_elements, n_bytes, blob in rows:
            shape = ast.literal_eval(shape_json)
            stats["raw_bytes"] += len(blob)

            if _eligible(list(shape), dtype):
                arr = to_float32(blob, dtype)
                data = _quantize_chunked(arr.reshape(-1), quantizer, pool)
                new_dtype, new_id = fmt, type_id
                stats["quantized"] += 1
            else:
                data = blob
                new_dtype, new_id = dtype, dtype_id
                stats["copied"] += 1

            stats["new_bytes"] += len(data)
            out.execute(
                INSERT_TENSOR,
                (name, shape_json, new_dtype, new_id, n_elements, len(data), data),
            )
            # Only when a terminal is watching. Piped, a carriage return does
            # not overwrite anything and every update survives as its own line.
            if verbose and sys.stdout.isatty():
                print(f"\r  {stats['quantized'] + stats['copied']:>4} tensors, "
                      f"{stats['new_bytes'] / 1e6:8.1f} MB written",
                      end="", flush=True)
    finally:
        pool.shutdown()

    out.execute("COMMIT")
    out.close()
    src.close()

    stats["seconds"] = time.perf_counter() - started
    stats["format"] = fmt
    stats["output"] = output_path
    if verbose:
        _report(stats)
    return stats


def _report(s: dict):
    raw, new = s["raw_bytes"], s["new_bytes"]
    if sys.stdout.isatty():
        print(f"\r{' ' * 48}\r", end="")
    print(f"Quantized to {s['format']}: {s['quantized']} tensors, "
          f"{s['copied']} copied through unchanged")
    print(f"  {raw / 1e6:.1f} MB -> {new / 1e6:.1f} MB "
          f"({100 * new / raw:.1f}% of the original)")
    print(f"  {s['seconds']:.1f}s")
    print(f"  {s['output']}")
