"""Getting a tensor out of the database and into a form that multiplies.

This is the layer between ``SELECT blob FROM tensors`` and the arithmetic:
it decides what a row becomes in memory, and it is where most of the load
time goes. A row can arrive as float32, as float16 left alone, as a GGML
quantized block repacked into the backend's own layout with no second
rounding, or as one expert lifted out of a stacked tensor -- and the choice
between those is what ``--pack``, ``--stream`` and the expert index are
each asking for.

Two things here are about latency rather than correctness. Prefetching
reads and unpacks the next layers on a thread pool while the current one
is being multiplied, so the disk and the CPU overlap; the condition
variable is what stops a consumer from reading a name that a producer is
still working on. And the memory map is turned on only when the database
is at most half of physical memory -- worth 7% on a file that fits, and
3.7x *slower* on one that does not.
"""

import ast
import os
import threading
from collections import OrderedDict

import numpy as np

from reminis.backend import best_group as _best_group
from reminis.backend import select as select_backend
from reminis.db import open_for_read
from reminis.dtypes import (
    dequantize_to_float32,
    is_float_dtype,
    is_quantized_dtype,
    to_float32_any,
)
from reminis.errors import UnsupportedModel
from reminis.expert_index import read_layouts as read_expert_layouts
from reminis.ggml_affine import AFFINE_GROUP, can_repack, nearest_bits
from reminis.ggml_affine import repack as ggml_repack
from reminis.packed_index import read_layouts as read_packed_layouts


# ---------------------------------------------------------------- weights


class _IndexReader:
    """Reads blocks out of the expert index on threads of its own.

    One connection per thread, because a sqlite connection is not something
    to share, and read-only because these never write. The point is not
    parallel decoding -- there is no decoding left to do -- but parallel
    waiting: several reads in flight at once keep the disk's queue full,
    and none of them hold the interpreter lock while they wait.
    """

    def __init__(self, db_path: str, workers: int = 4):
        from concurrent.futures import ThreadPoolExecutor

        self.path = db_path
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="reminis-expert")
        self._pending = {}
        self.prefetched = 0
        self.waited = 0

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = open_for_read(self.path, check_same_thread=False)
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
        self.conn = open_for_read(db_path)
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
            self._reader = _IndexReader(db_path, workers=reader_threads)

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
            conn = open_for_read(self.path, check_same_thread=False)
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
