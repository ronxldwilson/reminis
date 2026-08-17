"""Run a model straight out of its database.

Everything a forward pass needs is already in the file: the weights are rows
in ``tensors``, the hyperparameters and the tokenizer's vocabulary and merges
are rows in ``model_meta``. So this module loads no config, downloads nothing,
and imports neither torch nor llama.cpp -- it is numpy over the result of a
``SELECT``.

The point is not speed. It is that a reminis database is a *complete*,
self-contained model rather than an archive of one: if it can generate text,
then the conversion kept everything that mattered, and a merged or
rolled-back or delta-applied database can be checked by asking it to speak
rather than by comparing hashes.

Speed, measured rather than assumed, turns out to be less embarrassing than
expected -- 0.86-0.89x llama.cpp's CPU token generation across three model
sizes, and faster than it at prompt processing, since both end up in the
platform BLAS for the large matrix multiplies. Against llama.cpp on the GPU
it is 2.6-2.8x slower.

``--stream`` takes that further. In streaming mode no weight is ever cached:
every matrix multiplication in every layer re-reads its operand from SQLite
and throws it away, so peak memory is one layer rather than one model. It is
slow, and it demonstrates the thing the whole project is about -- that the
model is data in a database, paged in on demand, not a file that must fit in
RAM.

Scope, deliberately narrow and loudly enforced:

  * llama-family and qwen2 architectures from GGUF (rotary, RMSNorm, SwiGLU,
    grouped-query attention), which covers llama, qwen2, smollm, mistral
  * float weights -- F32, F16, BF16
  * quantized weights, unpacked at load through the ``gguf`` package: every
    K-quant and i-quant llama.cpp writes. Note what this is not -- the blocks
    become float16 in memory, so a quantized model becomes *runnable*, not
    *small*. Quantization saves the file and the download here, not the RAM.
  * a byte-level BPE tokenizer stored in the database

Anything else raises rather than approximates, because a forward pass that
guesses produces fluent-looking nonsense, which is worse than an error.
"""

import ast
import heapq
import math
import os
import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np

from reminis.backend import best_group as _best_group
from reminis.backend import select as select_backend
from reminis.dtypes import (
    dequantize_to_float32,
    is_float_dtype,
    is_quantized_dtype,
    to_float32_any,
)
from reminis import arch as arch_registry
from reminis.expert_index import read_layouts as read_expert_layouts
from reminis.packed_index import read_layouts as read_packed_layouts
from reminis.ggml_affine import AFFINE_GROUP, can_repack, nearest_bits
from reminis.ggml_affine import repack as ggml_repack

# Which architectures this can run, and what each does differently, lives
# in `arch.py`. Adding a model means adding an entry there rather than
# threading another special case through this file.
#
# The rotary layout that comes back with it is not cosmetic:
#   "norm" rotates adjacent pairs (0,1), (2,3), ...   -- llama, mistral
#   "neox" rotates halves,  i with i + head_dim/2     -- qwen2
SUPPORTED_ARCHS = {name: arch_registry.get(name).rope_style
                   for name in arch_registry.names()}


class UnsupportedModel(Exception):
    """The database holds a model this forward pass does not implement."""


# ---------------------------------------------------------------- weights


class _IndexReader:
    """Reads blocks out of the expert index on threads of its own.

    One connection per thread, because a sqlite connection is not something
    to share, and read-only because these never write. The point is not
    parallel decoding -- there is no decoding left to do -- but parallel
    waiting: several reads in flight at once keep the disk's queue full,
    and none of them hold the interpreter lock while they wait.
    """

    def __init__(self, db_path: str, workers: int = 4, mmap_size: int = 0):
        from concurrent.futures import ThreadPoolExecutor

        self.path = db_path
        self.mmap_size = mmap_size
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="reminis-expert")
        self._pending = {}
        self.prefetched = 0
        self.waited = 0

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.execute("PRAGMA query_only = 1")
            conn.execute(f"PRAGMA mmap_size = {self.mmap_size}")
            self._local.conn = conn
        return conn

    def _read(self, rowid: int) -> bytes:
        with self._conn().blobopen("expert_index", "block", rowid,
                                   readonly=True) as handle:
            return handle.read()

    def submit(self, key, rowid: int):
        if key not in self._pending:
            self._pending[key] = self._pool.submit(self._read, rowid)

    def take(self, key, rowid: int) -> bytes:
        """The bytes for one block, waiting for a prefetch if one is running."""
        future = self._pending.pop(key, None)
        if future is None:
            self.waited += 1
            return self._read(rowid)
        self.prefetched += 1
        return future.result()

    def close(self):
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        self._pool.shutdown(wait=True)
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


class WeightStore:
    """Tensor access backed by SQLite.

    Cached mode reads each tensor once and keeps the float32 copy. Streaming
    mode reads it every single time it is used and keeps nothing, which is
    the mode that makes the "the model lives in the database" claim literal.
    """

    # Weights worth keeping packed: the per-layer matrices, which are almost
    # all of a model's bytes and are only ever used as the right-hand side of
    # a matrix multiply. Embeddings are indexed row-wise rather than
    # multiplied, and norms are tiny, so neither is packed.
    _PACKABLE = (
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        # Mixture-of-experts weights are stacked 3-D tensors and are most of
        # such a model's bytes, so they matter more here than anything else.
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
        # A recurrent block names its projections differently, and on a model
        # that is three-quarters recurrent those names are most of the
        # weights: leaving them off this list left 11 GB of a 13.8 GB model
        # expanded to float16, which is the whole difference between fitting
        # on a 16 GB machine and not.
        "attn_qkv.weight", "attn_gate.weight", "ssm_out.weight",
    )

    # The embedding table is indexed rather than multiplied, but on a model
    # with tied weights it is also the output projection -- and there it is
    # the single largest read of every token. Packing it pays for the whole
    # output side, and the handful of rows an embedding lookup needs are
    # decoded individually.
    _PACKABLE_EMBED = ("token_embd.weight", "output.weight")

    def __init__(self, db_path: str, stream: bool = False, backend=None,
                 pack_bits=None, pack_group: int = 128,
                 reader_threads: int = 0):
        self.path = db_path
        self.stream = stream
        self.backend = backend or select_backend("inference")
        self.pack_bits = "native" if pack_bits == "compact" else pack_bits
        self.pack_compact = pack_bits == "compact"
        self.pack_group = pack_group
        self.packed = 0
        self.packed_native = 0
        # Routing-aware expert loading: how many decoded experts to keep, and
        # at what width. Off unless asked for.
        self.expert_cache_size = 0
        self.expert_bits = None
        self._expert_cache = OrderedDict()
        self._expert_shapes = {}
        self._blobs = {}
        self._prefetch_pool = None
        self._reader = None
        self._expert_layouts = {}
        self._expert_hits = 0
        self._expert_misses = 0
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA query_only = 1")
        self.conn.execute(f"PRAGMA mmap_size = {self._mmap_size(db_path)}")
        self._cache: dict[str, np.ndarray] = {}
        self.bytes_read = 0
        self.reads = 0
        self.dequantized = 0
        self._shapes = {
            name: (shape, dtype)
            for name, shape, dtype in self.conn.execute(
                "SELECT name, shape, dtype FROM tensors"
            )
        }
        # A materialized index over the expert weights, if this database has
        # one. It is derived from the tensors above and can be dropped, so
        # its absence is ordinary rather than an error.
        self._expert_layouts = read_expert_layouts(self.conn)
        # The same for the dense matrices, which is what makes loading a
        # quantized model a read rather than a decode. Consulted only when
        # packing was asked for, since it holds nothing else.
        self._packed_layouts = read_packed_layouts(self.conn)
        self.from_index = 0
        # The width the index was built at, when it is of one mind about it.
        # A run asking for a different one gets the index's, because
        # rebuilding it per run is the cost it exists to remove -- but it
        # gets told, since otherwise `--pack 4` would quietly produce
        # three-bit weights and report nothing.
        widths = {layout.bits for layout in self._packed_layouts.values()}
        self.index_bits = widths.pop() if len(widths) == 1 else None
        if self._expert_layouts and reader_threads:
            self._reader = _IndexReader(db_path, workers=reader_threads,
                                        mmap_size=self._mmap_size(db_path))

    @staticmethod
    def _mmap_size(db_path: str) -> int:
        """How much of the file to map, which is all of it or none of it.

        Mapping the file rather than copying each blob through sqlite's own
        buffer measured 4.1 GB/s to 6.7 GB/s on a 258 MB model, and it is
        free there because the whole thing fits in memory several times
        over.

        It stops being free when the file is larger than the machine. A
        mapped page that has been touched stays resident until the kernel
        decides otherwise, so on a 22.9 GB database with 16 GB of RAM the
        map competes for memory with the weights the model is trying to
        keep, and every read pays for it: measured 1.70 ms per expert block
        mapped against 0.46 ms unmapped, which is 3.7x the wrong way.

        So this maps the file when it comfortably fits and does not when it
        does not, rather than asking for 32 GB and hoping. Half of physical
        memory is the line, which is where the two measured points fall
        either side of: a 4.4 GB model on a 16 GB machine is 7% faster
        mapped, and a 22.9 GB one is 3.7x slower.
        """
        try:
            size = os.path.getsize(db_path)
            available = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, AttributeError):
            return 0
        return size if size * 2 <= available else 0

    def has(self, name: str) -> bool:
        return name in self._shapes

    # How many bytes of unpacked-but-not-yet-packed tensors to keep in
    # flight. Unpacking produces float32, four times what the packed result
    # will be and sixteen times what a 2-bit source occupied, so this is
    # the one place where reading ahead can cost more memory than the model
    # itself.
    #
    # The depth is chosen against the model rather than fixed, because the
    # two demands are in direct competition: on a model that leaves room to
    # spare, a couple of gigabytes of lookahead costs nothing and keeps the
    # threads fed; on one that nearly fills the machine, the same lookahead
    # is taken from the weights and the system compresses one to make room
    # for the other. A tenth of the file, within these bounds, keeps enough
    # in flight to matter without being the reason it does not fit.
    PREFETCH_MIN = 256_000_000
    PREFETCH_MAX = 2_000_000_000
    PREFETCH_HEADROOM = 1_500_000_000

    def start_prefetch(self, names, workers: int = 6):
        """Unpack these tensors on other threads, ahead of being asked for.

        Loading a quantized model is almost entirely `gguf`'s dequantize,
        which is numpy over the block layout and holds no interpreter lock
        while it runs -- measured at 3.2x on eight threads. The packing that
        follows is a GPU call and stays on the calling thread, so the two
        overlap: the processor unpacks the next tensors while the device
        packs this one.

        Ordering is a request rather than a promise. Whatever has not
        arrived when `get` asks for it is simply unpacked there and then, so
        this can never change what is loaded, only when.
        """
        if self.stream or not names:
            return
        from concurrent.futures import ThreadPoolExecutor

        try:
            size = os.path.getsize(self.path)
        except OSError:
            size = self.PREFETCH_MAX
        self.prefetch_bytes_limit = min(
            self.PREFETCH_MAX, max(self.PREFETCH_MIN, size // 10))
        # Anything the packed index already holds needs no unpacking, so
        # reading ahead for it would be work done twice and thrown away.
        self._prefetch_order = [n for n in names if n in self._shapes
                                and n not in self._packed_layouts]
        if not self._prefetch_order:
            return
        self._prefetch_done = {}
        self._prefetch_lock = threading.Lock()
        self._prefetch_ready = threading.Condition(self._prefetch_lock)
        self._prefetch_in_progress = set()
        self._prefetch_bytes = 0
        self._prefetch_pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="reminis-unpack")
        self._prefetch_local = threading.local()
        self._prefetch_stop = threading.Event()
        for name in self._prefetch_order:
            self._prefetch_pool.submit(self._unpack_one, name)

    def _prefetch_conn(self):
        conn = getattr(self._prefetch_local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.execute("PRAGMA query_only = 1")
            self._prefetch_local.conn = conn
        return conn

    def _unpack_one(self, name):
        """One tensor's blocks to float32, on a worker thread.

        Waits while the buffer is full rather than racing ahead: the point
        is to stay a few tensors in front of the forward pass, not to
        materialise the whole model at four bytes a weight.
        """
        if self._prefetch_stop.is_set():
            return
        try:
            with self._prefetch_lock:
                self._prefetch_in_progress.add(name)

            while True:
                with self._prefetch_lock:
                    if self._prefetch_bytes < self.prefetch_bytes_limit:
                        break
                if self._prefetch_stop.wait(0.02):
                    return

            row = self._prefetch_conn().execute(
                "SELECT shape, dtype, data FROM tensors WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                return
            shape, dtype, blob = row
            if not is_quantized_dtype(dtype):
                return
            dims = tuple(ast.literal_eval(shape))[::-1]

            if can_repack(dtype):
                if self.pack_bits != "native" or not self._packable(name):
                    return
                packed = ggml_repack(blob, dtype, dims)
                if packed is None:
                    return
                entry = ("repacked", packed, dims, len(blob))
                size = packed[0].nbytes + packed[1].nbytes + packed[2].nbytes
            else:
                arr = dequantize_to_float32(blob, dtype).reshape(dims)
                entry = ("float32", (arr, dtype), dims, len(blob))
                size = arr.nbytes

            with self._prefetch_ready:
                self._prefetch_done[name] = entry
                self._prefetch_bytes += size
                self._prefetch_ready.notify_all()
        except Exception:
            pass
        finally:
            with self._prefetch_ready:
                self._prefetch_in_progress.discard(name)
                self._prefetch_ready.notify_all()

    def take_prefetched(self, name):
        with self._prefetch_ready:
            if name in self._prefetch_in_progress and name not in self._prefetch_done:
                self._prefetch_ready.wait_for(
                    lambda: name not in self._prefetch_in_progress
                    or name in self._prefetch_done)
            entry = self._prefetch_done.pop(name, None)
            if entry is None:
                return None
            kind, payload, _, _ = entry
            if kind == "repacked":
                self._prefetch_bytes -= sum(p.nbytes for p in payload[:3])
            else:
                self._prefetch_bytes -= payload[0].nbytes
        return entry

    def stop_prefetch(self):
        pool = getattr(self, "_prefetch_pool", None)
        if pool is None:
            return
        self._prefetch_stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        with self._prefetch_lock:
            self._prefetch_done.clear()
            self._prefetch_bytes = 0
        self._prefetch_pool = None

    def get(self, name: str):
        """A tensor in the backend's compute dtype, row-major.

        reminis stores `shape` reversed relative to the data layout, which is
        GGUF's convention, so reversing it back is what turns the blob into
        the matrix the maths expects: a linear layer's weight comes out
        (out_features, in_features).
        """
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        # Already in the kernel's layout, sitting in the database: read the
        # bytes and hand them over. No block format is decoded, no float32
        # is materialised, and nothing is quantized a second time -- all of
        # that happened once, when the index was built.
        indexed = self._from_packed_index(name)
        if indexed is not None:
            return indexed

        # Already unpacked by a background thread: skip straight to the
        # narrowing and packing, which belong on this thread because they
        # are device calls.
        ready = (self.take_prefetched(name)
                 if getattr(self, "_prefetch_pool", None) else None)
        if ready is not None:
            kind, payload, dims, n_bytes = ready
            self.bytes_read += n_bytes
            self.reads += 1

            if kind == "repacked":
                words, scales, biases, bits = payload
                arr = self.backend.adopt_packed(
                    words, scales, biases, bits, AFFINE_GROUP, dims,
                    self.pack_compact)
                self.packed += 1
                self.packed_native += 1
            else:
                arr32, dtype = payload
                arr = self.backend.from_numpy(arr32)
                self.backend.eval(arr)
                del arr32
                bits = (nearest_bits(dtype) if self.pack_bits == "native"
                        else None)
                if isinstance(self.pack_bits, int):
                    bits = self.pack_bits
                if (bits is not None and self.backend.can_pack()
                        and self._packable(name)):
                    arr = self.backend.pack(
                        arr, bits, _best_group(dims[-1], self.pack_group),
                        self.pack_compact)
                    self.packed += 1
                self.dequantized += 1

            if not self.stream:
                self._cache[name] = arr
            return arr

        row = self.conn.execute(
            "SELECT shape, dtype, data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise UnsupportedModel(f"This model has no tensor named '{name}'")
        shape, dtype, blob = row
        dims = tuple(ast.literal_eval(shape))[::-1]

        native = self._native_pack(name, blob, dtype, dims)
        if native is not None:
            self.bytes_read += len(blob)
            self.reads += 1
            self.packed += 1
            if can_repack(dtype):
                self.packed_native += 1
            if not self.stream:
                self._cache[name] = native
            return native

        if is_float_dtype(dtype):
            arr = self.backend.from_bytes(blob, dtype, dims)
        elif is_quantized_dtype(dtype):
            # Quantized blocks are unpacked to float32 and then handed to the
            # backend, which narrows them to its compute dtype. This costs
            # more memory than the file did -- the whole point of quantizing
            # is undone -- and it is what makes a quantized model runnable at
            # all rather than a hard refusal.
            self.dequantized += 1
            arr = self.backend.from_numpy(
                dequantize_to_float32(blob, dtype).reshape(dims)
            )
            # Unpacking produces float32, and the backend narrows it to its
            # own compute dtype. On a lazy backend both would stay alive
            # until something forced the issue, so the float32 copy of every
            # weight in the model would accumulate. Force it here.
            self.backend.eval(arr)
        else:
            raise UnsupportedModel(
                f"'{name}' is stored as {dtype}, which is neither a float type "
                f"nor a quantization reminis can unpack."
            )
        if self._should_pack(name):
            arr = self.backend.pack(arr, self.pack_bits,
                                    _best_group(arr.shape[-1], self.pack_group),
                                    self.pack_compact)
            self.packed += 1

        self.bytes_read += len(blob)
        self.reads += 1
        if not self.stream:
            self._cache[name] = arr
        return arr

    def _from_packed_index(self, name: str):
        """One tensor out of the packed index, or None if it is not there.

        Only consulted when this run asked for packed weights: the index
        holds them at one width, and a run that wants floats wants the
        original tensors instead. The width the index was built at wins
        over the one asked for, since rebuilding it per run is the cost it
        exists to remove -- `reminis prepare --weights` is where that
        choice is made, and `--drop` is how it is unmade.
        """
        if self.pack_bits is None or not self._packed_layouts:
            return None
        layout = self._packed_layouts.get(name)
        if layout is None or not self.backend.can_pack():
            return None

        with self.conn.blobopen("packed_index", "block", layout.row_id,
                                readonly=True) as handle:
            raw = handle.read()
        words, scales, biases = layout.split(raw)
        arr = self.backend.adopt_packed(words, scales, biases, layout.bits,
                                        layout.group_size, layout.shape,
                                        compact=True)
        self.bytes_read += len(raw)
        self.reads += 1
        self.packed += 1
        self.from_index += 1
        if not self.stream:
            self._cache[name] = arr
        return arr

    def _should_pack(self, name: str) -> bool:
        return (
            isinstance(self.pack_bits, int)
            and self.backend.can_pack()
            and self._packable(name)
        )

    def _packable(self, name: str) -> bool:
        if name in self._PACKABLE_EMBED:
            return True
        return name.startswith("blk.") and name.endswith(self._PACKABLE)

    def _native_pack(self, name, blob, dtype, dims):
        """The stored blocks handed to the backend without being unpacked.

        Several GGML quantizations are already affine within each group of
        32, which is the form the backend's quantized matmul wants, so they
        can be rewritten into its layout by moving bits around. No weight is
        ever decoded, nothing is rounded a second time, and the result is
        bit-identical to unpacking -- verified against the gguf package's own
        dequantization, which is the implementation of record.
        """
        if self.pack_bits != "native" or not self.backend.can_pack():
            return None
        if not self._packable(name) or not is_quantized_dtype(dtype):
            return None

        if not can_repack(dtype):
            # No exact affine form -- Q6_K and friends use 16-weight
            # sub-blocks, the i-quants use codebooks. Leaving them as float16
            # would cost more memory than the file did, so they are
            # re-quantized to the nearest width instead. That rounds a second
            # time, which is why it is counted apart from the exact path.
            bits = nearest_bits(dtype)
            if bits is None:
                return None
            arr = self.backend.from_numpy(
                dequantize_to_float32(blob, dtype).reshape(dims)
            )
            return self.backend.pack(arr, bits, _best_group(dims[-1]),
                                     self.pack_compact)

        packed = ggml_repack(blob, dtype, dims)
        if packed is None:
            return None
        words, scales, biases, bits = packed
        return self.backend.adopt_packed(words, scales, biases, bits,
                                         AFFINE_GROUP, dims, self.pack_compact)

    def expert(self, name: str, index: int):
        """One expert out of a stacked tensor, read on its own.

        A mixture of experts stores all of them in a single 3-D tensor, and
        a token uses a handful. Loading the whole stack to use four of
        thirty-two reads eight times more than the forward pass needs --
        which is fine when the model fits in memory and ruinous when it does
        not.

        Each expert is a contiguous run of that blob, so reading one is a
        byte range: incremental blob I/O rather than fetching the tensor and
        throwing most of it away. Recently used experts are kept, because
        routing is stable enough across tokens for that to pay.
        """
        key = (name, index)
        cached = self._expert_cache.get(key)
        if cached is not None:
            self._expert_cache.move_to_end(key)
            self._expert_hits += 1
            return cached

        layout = self._expert_layouts.get(name)
        if layout is not None:
            return self._from_index(key, layout, index)

        rowid, dtype, shape, n_bytes = self._expert_meta(name)
        dims = tuple(ast.literal_eval(shape))[::-1]
        n_experts = dims[0]
        stride = n_bytes // n_experts

        handle = self._blobs.get(name)
        if handle is None:
            handle = self.conn.blobopen("tensors", "data", rowid, readonly=True)
            self._blobs[name] = handle
        raw = handle[index * stride:(index + 1) * stride]

        self.bytes_read += len(raw)
        self.reads += 1
        self._expert_misses += 1

        # Let the backend unpack the block format itself where it can: that
        # uploads the bytes the file holds rather than a float32 expansion
        # of them, and does the arithmetic on the device.
        native = getattr(self.backend, f"dequantize_{dtype.lower()}", None)
        if native is not None:
            arr = native(raw, dims[1:])
        else:
            arr = self.backend.from_numpy(
                to_float32_any(raw, dtype).reshape(dims[1:])
            )
        if self.expert_bits:
            arr = self.backend.pack(arr, self.expert_bits,
                                    _best_group(dims[-1]), True)
        self.backend.eval(arr if not hasattr(arr, "q") else arr.q)

        self._remember_expert(key, arr)
        return arr

    def prefetch_experts(self, wanted):
        """Start reading these experts, without waiting for any of them.

        The router names every expert a layer needs at once, and each one
        is an independent read of a few megabytes. Issued one at a time
        they serialise: the disk sits idle between them while the GPU sits
        idle waiting. Issued together they queue, which is the depth an SSD
        wants, and they overlap with whatever the GPU is doing -- the reads
        happen on other threads and sqlite releases the interpreter lock
        for the duration.
        """
        if self._reader is None:
            return
        for name, index in wanted:
            if (name, index) in self._expert_cache:
                continue
            layout = self._expert_layouts.get(name)
            if layout is not None:
                self._reader.submit((name, index), layout.row_id(index))

    def _from_index(self, key, layout, index: int):
        """One expert out of the materialized index.

        Everything expensive already happened when the index was built, so
        this is a primary-key seek, one read of a contiguous blob, and the
        memcpy that puts it where the kernel can reach it. On gpt-oss-20b
        that is the difference between 8.65 ms and 2.6 ms per expert.
        """
        if self._reader is not None:
            raw = self._reader.take(key, layout.row_id(index))
        else:
            with self.conn.blobopen("expert_index", "block",
                                    layout.row_id(index), readonly=True) as h:
                raw = h.read()
        self.bytes_read += len(raw)
        self.reads += 1
        self._expert_misses += 1

        words, scales, biases = layout.split(raw)
        arr = self.backend.adopt_packed(
            words, scales, biases, layout.bits, layout.group_size,
            (layout.rows, layout.cols), compact=True,
        )
        self._remember_expert(key, arr)
        return arr

    def _remember_expert(self, key, arr):
        self._expert_cache[key] = arr
        while len(self._expert_cache) > self.expert_cache_size:
            self._expert_cache.popitem(last=False)

    def _expert_meta(self, name: str):
        meta = self._expert_shapes.get(name)
        if meta is None:
            meta = self.conn.execute(
                "SELECT id, dtype, shape, n_bytes FROM tensors WHERE name = ?",
                (name,),
            ).fetchone()
            self._expert_shapes[name] = meta
        return meta

    def get_numpy(self, name: str) -> np.ndarray:
        """A tensor as plain float32 numpy, whatever the backend is.

        For the handful of values that are not part of the arithmetic on the
        hot path -- rotary scaling factors, say -- and are used to build
        tables in full precision.
        """
        row = self.conn.execute(
            "SELECT dtype, data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise UnsupportedModel(f"This model has no tensor named '{name}'")
        from reminis.dtypes import to_float32_any

        return to_float32_any(row[1], row[0])

    def drop(self, name: str):
        """Forget a cached tensor, once something else stands in for it."""
        self._cache.pop(name, None)

    def can_stream_experts(self) -> bool:
        """Whether byte-range reads are available for expert loading."""
        return hasattr(self.conn, "blobopen")

    def close(self):
        self.stop_prefetch()
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        for handle in self._blobs.values():
            try:
                handle.close()
            except Exception:
                pass
        self._blobs.clear()
        self.conn.close()




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


# ---------------------------------------------------------------- tokenizer


@lru_cache(maxsize=1)
def _byte_unicode_maps():
    """GPT-2's reversible byte-to-printable-character mapping.

    Byte-level BPE needs every one of the 256 byte values to be a character
    it can merge, and control characters and spaces are not usable, so GPT-2
    maps the unprintable ones into an unused Unicode range. This is why a
    space shows up as 'Ġ' in the stored vocabulary.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_to_char = {b: chr(b) for b in printable}
    spare = 0
    for b in range(256):
        if b not in byte_to_char:
            byte_to_char[b] = chr(256 + spare)
            spare += 1
    return byte_to_char, {c: b for b, c in byte_to_char.items()}


# GPT-2's pre-tokenizer splits text before BPE ever runs, and the split rules
# are part of the tokenizer: change them and the ids change. The originals are
# written with \p{L} and \p{N}, which Python's `re` does not have.
#
# `\w` is the closest thing available, and it is off by exactly one character:
# it counts underscore as a word character where \p{L} and \p{N} do not. That
# one character matters. Left uncorrected, an underscore matches neither the
# letter class nor the punctuation class, the pre-tokenizer skips it, and
# `and_underscores` silently loses a token relative to every other
# implementation. So the classes below are built to put it back:
#
#   _LETTER  \p{L}                       word characters that are not digits
#                                        or underscore
#   _OTHER   [^\s\p{L}\p{N}]             not space, letter or digit -- which
#                                        means underscore is explicitly in
_LETTER = r"[^\W\d_]"
_OTHER = r"(?:[^\s\w]|_)"
_PRETOKENIZERS = {
    # llama 3 and qwen 2 share this shape: digits in runs of at most three,
    # and a leading non-letter allowed to attach to a word.
    "llama-bpe": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        rf"|(?:[^\r\n\w]|_)?{_LETTER}+"
        r"|\d{1,3}"
        rf"| ?{_OTHER}+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    ),
    # The GPT-2 original, which SmolLM and most older BPE vocabularies use.
    "default": (
        r"'s|'t|'re|'ve|'m|'ll|'d"
        rf"| ?{_LETTER}+"
        r"| ?\d+"
        rf"| ?{_OTHER}+"
        r"|\s+(?!\S)"
        r"|\s+"
    ),
}
# gpt-4o's splitter needs Unicode case categories -- an uppercase run
# followed by a lowercase run, with a contraction attached to the word
# rather than split from it, so "It's" is one piece and not two. Python's
# `re` has no \p{Lu}, so this one is only available when the `regex`
# module is installed, and the tokenizer says so rather than quietly
# splitting differently from every other implementation.
_GPT4O_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# qwen 3.5's splitter, which is not qwen2's and not llama 3's. Two
# differences, and the first one matters on every number in every prompt:
#
#   digits    `\p{N}` matches one digit at a time, where llama 3 takes runs
#             of up to three. "2026" is four pieces here and two there, and
#             a model asked to do arithmetic on the wrong pieces answers
#             fluently and wrongly.
#   marks     `\p{M}` joins combining marks to the letter they modify, so a
#             decomposed accent stays with its base rather than splitting
#             off into the punctuation class.
#
# Like gpt-4o's, this needs Unicode general categories that Python's `re`
# does not have, so it is only available with the `regex` module and says
# so rather than silently splitting a different way.
_QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

_PRETOKENIZERS["qwen2"] = _PRETOKENIZERS["llama-bpe"]
_PRETOKENIZERS["smollm"] = _PRETOKENIZERS["default"]
_PRETOKENIZERS["gpt-2"] = _PRETOKENIZERS["default"]


def build_tokenizer(meta: dict):
    """The tokenizer this model was trained with, rebuilt from the database.

    Two families cover everything reminis runs. ``gpt2`` is byte-level BPE,
    driven by an ordered merge list. ``llama`` is SentencePiece, which has no
    merge list at all -- it merges by a score attached to each token, so the
    two share almost no machinery despite both being called BPE.
    """
    model = meta.get("tokenizer.ggml.model")
    if model == "gpt2":
        return BPETokenizer(meta)
    if model in ("llama", "spm"):
        # SentencePiece proper merges by score. Some conversions keep the
        # SentencePiece alphabet but carry a merge list instead, and fill
        # every score with the same placeholder -- which is a merge-ranked
        # BPE wearing a SentencePiece vocabulary, and has to be encoded as
        # one. A single distinct score means the scores decide nothing.
        if _scores_are_inert(meta) and _parse_array(meta, "tokenizer.ggml.merges"):
            return PieceBPETokenizer(meta)
        return SPMTokenizer(meta)
    if model in ("gemma4", "gemma3", "gemma2", "gemma"):
        return PieceBPETokenizer(meta)
    raise UnsupportedModel(
        f"This model's tokenizer is '{model or 'missing'}'. reminis run "
        f"implements byte-level BPE ('gpt2'), SentencePiece ('llama') and "
        f"merge-ranked SentencePiece ('gemma')."
    )


def _scores_are_inert(meta: dict) -> bool:
    """Whether the token scores carry no information.

    A conversion that drops SentencePiece's scores writes the same
    placeholder for every token. Merging by them would then be merging by
    nothing, in whatever order the vocabulary happens to be in.
    """
    try:
        scores = _numeric_array(meta, "tokenizer.ggml.scores")
    except Exception:
        return False
    return bool(scores) and len(set(scores)) == 1


def _numeric_array(meta: dict, key: str) -> list:
    """A GGUF numeric array, which arrives as one-element lists per entry."""
    return [
        v[0] if isinstance(v, (list, tuple)) else v
        for v in _parse_array(meta, key)
    ]


class SPMTokenizer:
    """SentencePiece, as llama.cpp implements it.

    There is no merge list. Every token carries a score, and the algorithm
    repeatedly merges whichever adjacent pair forms the highest-scoring token
    in the vocabulary -- so the vocabulary itself encodes the merge order.
    Text is pre-escaped by replacing spaces with U+2581, which is why a
    leading space appears in almost every token.

    Anything with no token at all falls back to one token per *byte*, which
    is why the vocabulary contains 256 entries named `<0x00>` through
    `<0xFF>`.
    """

    SPACE = "\u2581"

    def __init__(self, meta: dict):
        self.tokens = _parse_array(meta, "tokenizer.ggml.tokens")
        self.scores = _numeric_array(meta, "tokenizer.ggml.scores")
        types = _numeric_array(meta, "tokenizer.ggml.token_type")
        if not self.tokens or not self.scores:
            raise UnsupportedModel(
                "The database has no SentencePiece vocabulary in it."
            )

        self.ids = {t: i for i, t in enumerate(self.tokens)}
        # Type 6 is BYTE: the fallback for text with no token of its own.
        self.byte_ids = {}
        for i, t in enumerate(types):
            if int(t) == 6:
                text = self.tokens[i]
                if len(text) == 6 and text.startswith("<0x"):
                    self.byte_ids[int(text[3:5], 16)] = i
        self.id_to_byte = {i: b for b, i in self.byte_ids.items()}

        self.specials = sorted(
            (self.tokens[i] for i, t in enumerate(types) if int(t) in (3, 4)),
            key=len, reverse=True,
        )
        import re

        self._re = re
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(s) for s in self.specials) + ")")
            if self.specials else None
        )

        self.bos_id = _int_or_none(meta.get("tokenizer.ggml.bos_token_id"))
        self.eos_id = _int_or_none(meta.get("tokenizer.ggml.eos_token_id"))
        # SentencePiece models add the beginning-of-text token by default,
        # and prepend a space so the first word looks like any other.
        self.add_bos = str(
            meta.get("tokenizer.ggml.add_bos_token", "True")
        ).lower() == "true"
        self.add_space_prefix = str(
            meta.get("tokenizer.ggml.add_space_prefix", "True")
        ).lower() == "true"
        self.chat_template = meta.get("tokenizer.chat_template", "")

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = []
        if add_special and self.add_bos and self.bos_id is not None:
            ids.append(self.bos_id)

        chunks = self._special_pattern.split(text) if self._special_pattern else [text]
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.ids and chunk in self.specials:
                ids.append(self.ids[chunk])
                continue
            ids.extend(self._encode_piece(chunk))
        return ids

    def _encode_piece(self, text: str) -> list[int]:
        # Every stretch of ordinary text gets the leading space, not just the
        # first: "x[INST]y" encodes its "y" as "_y", the same as its "x".
        # Checked against llama.cpp, which does the same.
        if self.add_space_prefix:
            text = " " + text
        text = text.replace(" ", self.SPACE)
        if not text:
            return []

        chars = list(text)
        n = len(chars)
        # A doubly linked list over the characters, so merging is a pointer
        # update rather than a rebuild of the sequence.
        length = [1] * n
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[-1] = -1

        heap = []

        def consider(left, right):
            if left == -1 or right == -1:
                return
            piece = "".join(chars[left:left + length[left] + length[right]])
            token = self.ids.get(piece)
            if token is None:
                return
            # A min-heap over (-score, left) is a max-heap over score that
            # breaks ties toward the earlier position, which is the order
            # llama.cpp's priority queue produces.
            heapq.heappush(heap, (-self.scores[token], left, right, len(piece)))

        for i in range(1, n):
            consider(i - 1, i)

        while heap:
            _, left, right, size = heapq.heappop(heap)
            if length[left] == 0 or length[right] == 0:
                continue
            if length[left] + length[right] != size:
                continue

            length[left] += length[right]
            length[right] = 0
            nxt[left] = nxt[right]
            if nxt[right] != -1:
                prev[nxt[right]] = left

            consider(prev[left], left)
            consider(left, nxt[left])

        out = []
        i = 0
        while i != -1:
            self._emit("".join(chars[i:i + length[i]]), out)
            i = nxt[i]
        return out

    def _emit(self, piece, out):
        """One surviving run to ids, or to its raw bytes if it has no token.

        A run only ever grew by merging into a string the vocabulary
        contains, so anything left without a token is a single character the
        model has never seen -- and those become one token per UTF-8 byte.
        """
        token = self.ids.get(piece)
        if token is not None:
            out.append(token)
            return
        for byte in piece.encode("utf-8"):
            if byte in self.byte_ids:
                out.append(self.byte_ids[byte])

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for i in ids:
            if i in self.id_to_byte:
                out.append(self.id_to_byte[i])
                continue
            if 0 <= i < len(self.tokens):
                out += self.tokens[i].replace(self.SPACE, " ").encode("utf-8")
        text = out.decode("utf-8", errors="replace")
        return text[1:] if self.add_space_prefix and text.startswith(" ") else text

    def decode_one(self, token_id: int) -> str:
        if token_id in self.id_to_byte:
            return bytes([self.id_to_byte[token_id]]).decode("utf-8", errors="replace")
        if 0 <= token_id < len(self.tokens):
            return self.tokens[token_id].replace(self.SPACE, " ")
        return ""


class BPETokenizer:
    """Byte-level BPE, rebuilt from the vocabulary and merges in the database."""

    def __init__(self, meta: dict):
        import re

        self.tokens = _parse_array(meta, "tokenizer.ggml.tokens")
        merges = _parse_array(meta, "tokenizer.ggml.merges")
        if not self.tokens or not merges:
            raise UnsupportedModel(
                "The database has no tokenizer vocabulary in it, so there is "
                "nothing to encode with."
            )

        self.ids = {t: i for i, t in enumerate(self.tokens)}
        self.ranks = {tuple(m.split(" ")): i for i, m in enumerate(merges)}

        # Token types 3 and 4 are CONTROL and USER_DEFINED -- <|im_start|>
        # and friends, and markers like <think> that a chat template writes
        # as literal text. Both must be matched whole rather than run
        # through the splitter, which would cut <think> at the boundary
        # between its punctuation and its letters and leave BPE unable to
        # rebuild it: the pieces are all real tokens, so nothing complains,
        # and the model is handed a sequence it was never trained on.
        #
        # Only CONTROL was collected here, while the SentencePiece
        # tokenizer above already took both. That difference was the bug
        # rather than a distinction being drawn.
        #
        # Numeric GGUF arrays come back as one-element lists per entry,
        # since that is how the reader hands them over.
        types = [
            t[0] if isinstance(t, (list, tuple)) else t
            for t in _parse_array(meta, "tokenizer.ggml.token_type")
        ]
        self.specials = sorted(
            (self.tokens[i] for i, t in enumerate(types) if int(t) in (3, 4)),
            key=len, reverse=True,
        )

        pre = meta.get("tokenizer.ggml.pre", "default")
        if pre in ("gpt-4o", "qwen35"):
            unicode_pattern = _GPT4O_PATTERN if pre == "gpt-4o" else _QWEN35_PATTERN
            try:
                import regex
            except ImportError:
                raise UnsupportedModel(
                    f"This model uses the {pre} pre-tokenizer, which needs "
                    f"Unicode general categories that Python's `re` does not "
                    f"have.\n"
                    f"  pip install regex"
                )
            self.pattern = regex.compile(unicode_pattern)
        else:
            self.pattern = re.compile(
                _PRETOKENIZERS.get(pre, _PRETOKENIZERS["default"])
            )
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(s) for s in self.specials) + ")")
            if self.specials else None
        )
        self._re = re

        self.bos_id = _int_or_none(meta.get("tokenizer.ggml.bos_token_id"))
        self.eos_id = _int_or_none(meta.get("tokenizer.ggml.eos_token_id"))
        self.add_bos = str(meta.get("tokenizer.ggml.add_bos_token", "False")).lower() == "true"
        self.chat_template = meta.get("tokenizer.chat_template", "")

        self._byte_to_char, self._char_to_byte = _byte_unicode_maps()
        self._bpe_cache: dict[str, list[str]] = {}

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = []
        if add_special and self.add_bos and self.bos_id is not None:
            ids.append(self.bos_id)

        chunks = (
            self._special_pattern.split(text) if self._special_pattern else [text]
        )
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.ids and chunk in self.specials:
                ids.append(self.ids[chunk])
                continue
            for piece in self.pattern.findall(chunk):
                encoded = "".join(self._byte_to_char[b] for b in piece.encode("utf-8"))
                for token in self._bpe(encoded):
                    token_id = self.ids.get(token)
                    if token_id is None:
                        # Every single byte is in the vocabulary, so falling
                        # back to bytes always terminates.
                        ids.extend(self.ids[c] for c in token if c in self.ids)
                    else:
                        ids.append(token_id)
        return ids

    def _bpe(self, word: str) -> list[str]:
        cached = self._bpe_cache.get(word)
        if cached is not None:
            return cached

        parts = list(word)
        while len(parts) > 1:
            pairs = zip(parts, parts[1:])
            best, best_rank = None, None
            for i, pair in enumerate(pairs):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = i, rank
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]

        self._bpe_cache[word] = parts
        return parts

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.tokens[i] for i in ids if 0 <= i < len(self.tokens))
        raw = bytes(self._char_to_byte.get(c, ord("?") if ord(c) > 255 else ord(c))
                    for c in text)
        return raw.decode("utf-8", errors="replace")

    def decode_one(self, token_id: int) -> str:
        return self.decode([token_id])


class PieceBPETokenizer(BPETokenizer):
    """BPE over a SentencePiece vocabulary, as Gemma's is stored.

    It is BPE -- an ordered merge list decides everything -- but the
    alphabet is SentencePiece's rather than GPT-2's, and the difference is
    not cosmetic:

      * a space is the character `▁`, not GPT-2's `Ġ`, and the merges are
        written in terms of it;
      * a byte with no piece of its own appears as the literal text
        `<0xE2>` rather than as a character in a remapped Unicode range.

    So the byte-level mapping the parent class applies is exactly wrong
    here, and both directions are overridden. What is inherited is the part
    that is genuinely shared: merge ranks, the merge loop, and the handling
    of control tokens.
    """

    SPACE = "▁"

    def __init__(self, meta: dict):
        super().__init__(meta)
        # A vocabulary this size is worth one pass to find the byte pieces,
        # rather than formatting a string per byte on every fallback.
        self._byte_piece = {}
        for value in range(256):
            piece = f"<0x{value:02X}>"
            if piece in self.ids:
                self._byte_piece[value] = piece
        self._piece_byte = {p: v for v, p in self._byte_piece.items()}
        # SentencePiece models differ on whether an implicit space is added
        # in front of the text; Gemma's says not to.
        self.add_space_prefix = str(
            meta.get("tokenizer.ggml.add_space_prefix", "True")
        ).lower() == "true"

    def _split(self, text: str) -> list[str]:
        """Words, each carrying its own leading space.

        Merging the whole text at once would be quadratic in its length and
        would also let a merge straddle a space that no training example
        ever presented as one piece. Splitting before each space marker is
        what SentencePiece itself does.
        """
        parts, current = [], ""
        for char in text:
            if char == self.SPACE and current:
                parts.append(current)
                current = char
            else:
                current += char
        if current:
            parts.append(current)
        return parts

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = []
        if add_special and self.add_bos and self.bos_id is not None:
            ids.append(self.bos_id)

        chunks = (
            self._special_pattern.split(text) if self._special_pattern else [text]
        )
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.ids and chunk in self.specials:
                ids.append(self.ids[chunk])
                continue
            marked = chunk.replace(" ", self.SPACE)
            if self.add_space_prefix and not marked.startswith(self.SPACE):
                marked = self.SPACE + marked
            for word in self._split(marked):
                for token in self._bpe(word):
                    token_id = self.ids.get(token)
                    if token_id is not None:
                        ids.append(token_id)
                        continue
                    # No piece for this text: spell it out a byte at a
                    # time, which the vocabulary always has room for.
                    for value in token.encode("utf-8"):
                        piece = self._byte_piece.get(value)
                        if piece is not None:
                            ids.append(self.ids[piece])
        return ids

    def decode(self, ids: list[int]) -> str:
        # Byte pieces have to be gathered into runs before decoding, since a
        # multi-byte character is spelled as several of them and neither
        # half is valid UTF-8 alone.
        out, pending = [], bytearray()

        def flush():
            if pending:
                out.append(pending.decode("utf-8", errors="replace"))
                pending.clear()

        for i in ids:
            if not 0 <= i < len(self.tokens):
                continue
            piece = self.tokens[i]
            value = self._piece_byte.get(piece)
            if value is not None:
                pending.append(value)
                continue
            flush()
            out.append(piece.replace(self.SPACE, " "))
        flush()
        return "".join(out)


def _parse_array(meta: dict, key: str) -> list:
    """Read one of the list-shaped metadata values back into a Python list.

    The converter stores GGUF arrays as their Python repr, so this is the
    inverse. `literal_eval` rather than `eval`: these strings come out of a
    file someone downloaded.
    """
    raw = meta.get(key)
    if not raw:
        return []
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise UnsupportedModel(f"Could not read {key} from this database: {exc}")


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
        self._rope_cache: tuple | None = None
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

    # -- rotary embeddings ------------------------------------------------

    def _rope_tables(self, n_positions: int, offset: int):
        """cos/sin for positions [offset, offset + n_positions).

        Every layer in a forward pass asks for the same positions, so the
        trigonometry was being redone once per layer -- thirty times per
        token on a 135M model. One entry is enough to hold it.
        """
        if self._rope_cache is not None:
            key_n, key_offset, cos, sin = self._rope_cache
            if key_n == n_positions and key_offset == offset:
                return cos, sin

        cfg = self.cfg
        half = cfg.rope_dim // 2
        # The tables are built in numpy at full precision regardless of the
        # backend. They are tiny, they are cached, and the angles are the one
        # place where half precision would visibly cost accuracy: an error in
        # a position's angle shifts every token that attends to it.
        inv_freq = 1.0 / (
            cfg.rope_base ** (np.arange(half, dtype=np.float32) * 2.0 / cfg.rope_dim)
        )
        if cfg.rope_factors is not None:
            inv_freq = inv_freq / cfg.rope_factors[:half]
        pos = np.arange(offset, offset + n_positions, dtype=np.float32)
        angles = pos[:, None] * inv_freq[None, :]
        cos = self.backend.from_numpy(np.cos(angles))
        sin = self.backend.from_numpy(np.sin(angles))
        self._rope_cache = (n_positions, offset, cos, sin)
        return cos, sin

    def _apply_rope(self, x, cos, sin):
        """Rotate the first `rope_dim` channels of each head.

        Two layouts exist and they are not interchangeable. llama.cpp's
        "norm" rotates adjacent channels as pairs; "neox" rotates channel i
        against channel i + rope_dim/2. Which one a model wants depends on
        how its weights were laid out at conversion time, so it comes from
        the architecture rather than from a guess.
        """
        d = self.cfg.rope_dim
        rot, rest = x[..., :d], x[..., d:]
        cos = cos[:, None, :]
        sin = sin[:, None, :]

        xp = self.backend.xp
        if self.cfg.rope_style == "norm":
            even, odd = rot[..., 0::2], rot[..., 1::2]
            # Interleaving the two halves back together by stacking and
            # reshaping rather than by assigning into a buffer, since not
            # every backend's arrays can be written into piecewise.
            rotated = xp.stack([even * cos - odd * sin, even * sin + odd * cos],
                               axis=-1)
            out = rotated.reshape(rot.shape)
        else:
            half = d // 2
            first, second = rot[..., :half], rot[..., half:]
            out = xp.concatenate(
                [first * cos - second * sin, first * sin + second * cos], axis=-1
            )

        return out if rest.shape[-1] == 0 else xp.concatenate([out, rest], axis=-1)

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


class KVCache:
    """Keys and values for every layer.

    Growing this with `np.concatenate` reallocates and recopies the entire
    cache on every token, which turns a linear cost into a quadratic one and
    is invisible until the context is long. Given a capacity up front it
    allocates once and writes each token into place, returning a view.
    Without one it still grows, in doubling steps rather than by one.
    """

    def __init__(self, n_layers: int, capacity: int | None = None, backend=None,
                 quantize_bits: int | None = None):
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
        if layer == len(self.k) - 1:
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

        if layer == len(self.k) - 1:
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


# ---------------------------------------------------------------- sampling


def _sample(logits: np.ndarray, temperature: float, top_p: float, rng) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))

    scaled = logits.astype(np.float32) / np.float32(temperature)
    scaled = scaled - np.max(scaled)
    np.exp(scaled, out=scaled)
    probs = scaled / np.sum(scaled)

    if 0 < top_p < 1:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        # Keep the smallest set of tokens whose mass reaches top_p, always
        # including the first one so the set is never empty.
        keep = int(np.searchsorted(cumulative, top_p) + 1)
        chosen = order[:keep]
        renormalised = probs[chosen] / probs[chosen].sum()
        return int(rng.choice(chosen, p=renormalised))

    return int(rng.choice(len(probs), p=probs))


def generate(
    db_path: str,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int | None = None,
    stream: bool = False,
    chat: bool = False,
    think: bool = False,
    stop_at_eos: bool = True,
    verbose: bool = True,
    on_token=None,
    backend: str | None = None,
    pack_bits=None,
    kv_bits: int | None = None,
    expert_cache: int = 0,
    expert_bits: int | None = None,
    preload: bool = False,
) -> dict:
    """Generate text from a model stored in a reminis database.

    Args:
        db_path: The model database.
        prompt: The prompt text.
        max_tokens: How many tokens to generate at most.
        temperature: 0 is greedy; higher is more random.
        top_p: Nucleus sampling cutoff. 1 disables it.
        seed: Seed for sampling, so a run can be repeated exactly.
        kv_bits: Compress the key/value cache to this many bits. The cache
            grows with the context where the weights do not, so this is
            what decides whether a long prompt fits. It costs speed rather
            than saving it -- there is no quantized attention kernel, so
            the cache is decompressed to attend.
        pack_bits: Keep the big per-layer matrices packed at this many bits
            instead of unpacking them to float16, on a backend that can
            multiply them packed. Trades accuracy for memory: 6 keeps the
            top-5 ranking intact for 1.7x less, 4 goes to 2.1x less and
            visibly reorders it.
        stream: Re-read every weight from SQLite instead of caching it, so
            peak memory is one layer rather than the whole model.
        chat: Wrap the prompt in the model's chat template, when it has a
            ChatML one.
        think: On a reasoning model, leave its thinking channel open rather
            than closing it immediately. The answer then arrives after the
            working, which needs a token budget to match.
        stop_at_eos: Stop when the model emits its end-of-text token.
        verbose: Print the header, the prompt, and the closing timings. The
            generated text itself is printed either way, so that piping the
            output somewhere gives the completion and nothing else.
        on_token: Optional callback, called with each decoded token string.
            Supplying one turns off printing, since the caller is handling it.

    Returns:
        A dict with the prompt, the completion, and timing.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    chosen = select_backend("inference", backend)
    model = Model(db_path, stream=stream, backend=chosen, pack_bits=pack_bits,
                  expert_cache=expert_cache, expert_bits=expert_bits)
    if preload:
        if verbose:
            print("Loading the expert index into memory", end="", flush=True)
        loaded = model.preload_experts()
        if verbose:
            print(f"\r{loaded['experts']:,} experts resident "
                  f"({loaded['bytes'] / 1e9:.2f} GB) in "
                  f"{loaded['seconds']:.0f}s")
    tok = model.tokenizer
    rng = np.random.default_rng(seed)

    try:
        text = _apply_chat_template(prompt, tok, think) if chat else prompt
        tokens = tok.encode(text)
        if not tokens:
            raise ValueError("The prompt encoded to zero tokens")
        if len(tokens) >= model.cfg.context_length:
            raise ValueError(
                f"The prompt is {len(tokens)} tokens and this model's context "
                f"is {model.cfg.context_length}"
            )

        if verbose:
            mode = "streaming from SQLite" if stream else "weights cached in RAM"
            print(f"{model.meta.get('general.name', Path(db_path).name)} "
                  f"| {model.cfg.arch} | {model.cfg.n_layers} layers | "
                  f"{chosen.describe()} | {mode}")
            print(f"{len(tokens)} prompt tokens\n")
            print(text, end="", flush=True)

        if pack_bits is not None and not chosen.can_pack():
            print(f"Note: the {chosen.name} backend cannot multiply packed "
                  f"weights, so --pack was ignored.")

        # A packed index holds one width. Honouring `--pack` against it would
        # mean rebuilding every weight, which is the work the index exists to
        # have already done -- so the index wins and says so. Silence here
        # would mean asking for four bits and running three.
        index_bits = model.store.index_bits
        if index_bits is not None and pack_bits is not None:
            asked = pack_bits if isinstance(pack_bits, int) else None
            if asked != index_bits:
                wanted = (f"--pack {asked}" if asked is not None
                          else "bit-exact packing")
                print(f"Note: this database has a packed index built at "
                      f"{index_bits} bits, and it is being used. {wanted} "
                      f"would mean rebuilding every weight, which is what "
                      f"the index is there to avoid.\n"
                      f"      To run at another width: reminis prepare "
                      f"<db> --weights --bits N   (or --weights --drop)")

        cache = KVCache(model.cfg.n_layers, capacity=len(tokens) + max_tokens,
                        backend=chosen, quantize_bits=kv_bits)

        t0 = time.time()
        logits = model.forward(tokens, cache, offset=0)
        prefill_seconds = time.time() - t0

        if (pack_bits is not None and chosen.can_pack()
                and model.store.packed == 0):
            # Bit-exact packing rearranges quantization blocks, and a float
            # model has none to rearrange. Saying so beats leaving someone to
            # wonder why a flag they passed changed nothing.
            print(
                f"Note: --pack {pack_bits} had nothing to pack. This model's "
                f"weights are stored as floats, and bit-exact packing only "
                f"rearranges existing quantization blocks.\n"
                f"      Use --pack 8 to quantize them instead."
            )

        produced = []
        t1 = time.time()
        for _ in range(max_tokens):
            token_id = _sample(logits, temperature, top_p, rng)
            if stop_at_eos and tok.eos_id is not None and token_id == tok.eos_id:
                break
            produced.append(token_id)
            piece = tok.decode_one(token_id)
            if on_token:
                on_token(piece)
            else:
                print(piece, end="", flush=True)
            logits = model.forward([token_id], cache, offset=cache.length)

        decode_seconds = time.time() - t1
        completion = tok.decode(produced)

        if not on_token:
            print()
        if verbose:
            rate = len(produced) / decode_seconds if decode_seconds else 0
            print(f"\n{len(tokens)} prompt tokens in {prefill_seconds:.2f}s, "
                  f"{len(produced)} generated in {decode_seconds:.2f}s "
                  f"({rate:.1f} tok/s)")
            read = model.store.bytes_read / (1024 ** 2)
            if stream:
                print(f"Read {read:,.0f} MB from SQLite across "
                      f"{model.store.reads:,} queries, cached nothing")
            else:
                print(f"Read {read:,.0f} MB from SQLite once, then reused it")

        return {
            "prompt": text,
            "completion": completion,
            "prompt_tokens": len(tokens),
            "generated_tokens": len(produced),
            "token_ids": produced,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "bytes_read": model.store.bytes_read,
            "queries": model.store.reads,
            "backend": chosen.name,
            "packed_tensors": model.store.packed,
        }
    finally:
        model.close()


def _apply_chat_template(prompt: str, tok, think: bool = False) -> str:
    """Wrap a prompt as a chat turn, for models that use ChatML.

    The stored template is Jinja, and rendering Jinja to run one prompt is
    more machinery than it is worth. ChatML is recognisable and it is what
    the small instruct models here use; anything else is left alone rather
    than mangled into a format the model was not trained on.

    A reasoning model needs one thing more. Its template does not stop at
    the assistant marker -- it opens a thinking channel too, and the model
    was trained expecting to find one already open:

        <|im_start|>assistant\\n<think>\\n

    Left off, the model opens the channel itself and reasons for as long as
    it likes before answering, so a run with any sane token budget shows
    working and no answer. That reads exactly like a broken forward pass
    and is not one. Closing the channel immediately -- an empty pair of
    tags -- is how these templates express "answer directly", and it is
    the default here because a one-shot completion is the case `run`
    serves. `--think` asks for the channel left open.
    """
    template = tok.chat_template or ""
    if "<|im_start|>" not in template:
        return prompt
    turn = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if "<think>" not in template:
        return turn
    return turn + ("<think>\n" if think else "<think>\n\n</think>\n\n")


def run_cli(args, on_error=None):
    """Entry point for `reminis run`, kept here so the CLI stays thin."""
    try:
        generate(
            args.input, args.prompt,
            max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p,
            seed=args.seed, stream=args.stream, chat=args.chat,
            think=getattr(args, 'think', False),
            verbose=not args.quiet, backend=args.backend,
            pack_bits=args.pack, kv_bits=args.kv_bits,
            expert_cache=0 if args.experts in (None, "all") else args.experts,
            preload=args.experts == "all",
        )
    except (UnsupportedModel, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
