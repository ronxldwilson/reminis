"""A materialized index over a mixture-of-experts model's weights.

A mixture of experts is the one architecture where the read pattern is
decided at query time. The router looks at a token and names four experts
out of thirty-two, so a forward pass touches an eighth of the weights and
cannot say which eighth until it gets there. Reading only those four is
what lets a 12 GB model run on a machine with 16 GB of RAM, and it is the
thing a memory-mapped file cannot do on purpose -- mmap faults in whatever
the pointer happened to touch, after the fact.

Reading the right bytes turned out not to be the expensive part. Measured
on gpt-oss-20b, fetching one expert cost 8.65 ms, of which the actual read
was 2.5 ms and the other 6.1 ms was arithmetic: unpacking MXFP4's 17-byte
blocks and re-packing them into the layout the quantized matmul kernel
wants. Every token paid that for about 116 experts, and every token paid it
again for the same experts, because the result was thrown away when the
cache evicted it.

That work is a pure function of bytes that are already in the database, so
it belongs in the database. This builds a second physical representation of
the expert weights -- already unpacked, already re-quantized, already in
the kernel's layout, one row per expert, clustered so that consecutive
experts of the same tensor are consecutive on disk. It is an index in the
ordinary sense: redundant, derived, ordered for one access path, and
droppable without losing anything.

    reminis prepare model.db          build it
    reminis prepare model.db --drop   throw it away

With it, fetching an expert is a primary-key seek and a memcpy. The 6.1 ms
of arithmetic is paid once, at build time, instead of a few hundred times
per second forever.
"""

import ast
import sqlite3
import time

from reminis import blockindex
from reminis.db import mmap_bytes

# The suffixes of the stacked 3-D expert tensors. Everything else in a
# mixture-of-experts model -- the router, the per-expert biases, attention --
# is small enough to hold whole and is left alone.
EXPERT_TENSORS = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    # Gemma stores the gate and up projections of each expert as one
    # matrix of twice the width. Leaving it out of this list indexed half
    # of such a model and left the larger half being decoded per token,
    # which is the cost the index exists to remove.
    "ffn_gate_up_exps.weight",
    "ffn_down_exps.weight",
)

# 4 bits at a group size of 128 costs 4.25 bits per weight once the scale
# and bias are counted, which is what MXFP4 costs in the file. So the index
# is the same size as the tensors it indexes, and the choice is free.
DEFAULT_BITS = 4
DEFAULT_GROUP = 128

TABLE = "expert_index"
SCHEMA = blockindex.schema(
    TABLE,
    "    base_id    INTEGER NOT NULL,\n    n_experts  INTEGER NOT NULL,",
)


class Layout(blockindex.BlockLayout):
    """Where one tensor's experts live, and how to read one back.

    A row is one expert of a stacked tensor, so the identity is the
    `base_id` the stack starts at and how many experts follow it.
    """

    __slots__ = ("base_id", "n_experts")
    COLUMNS = ("tensor", "base_id", "n_experts") + blockindex.GEOMETRY_COLUMNS

    def row_id(self, expert: int) -> int:
        return self.base_id + expert


def read_layouts(conn: sqlite3.Connection) -> dict:
    """Every indexed tensor, or an empty mapping if there is no index."""
    return blockindex.read_layouts(conn, TABLE, Layout)


def expert_tensor_names(conn: sqlite3.Connection) -> list:
    """The stacked expert tensors in this model, in layer order.

    Ordered so that the ids assigned below are clustered the way the
    forward pass reads them: a layer's three projections together, and
    within each, the experts in index order.
    """
    names = [
        name for (name,) in conn.execute("SELECT name FROM tensors")
        if name.startswith("blk.") and name.endswith(EXPERT_TENSORS)
    ]

    def order(name):
        parts = name.split(".")
        return (int(parts[1]), EXPERT_TENSORS.index(".".join(parts[2:])))

    return sorted(names, key=order)


def build(db_path: str, backend=None, bits: int = DEFAULT_BITS,
          group_size: int = DEFAULT_GROUP, progress=None) -> dict:
    """Build the index, replacing any existing one.

    Returns a summary: how many experts were written, how many bytes, and
    how long it took.
    """
    from reminis.backend import best_group, select as select_backend
    from reminis.dtypes import to_float32_any

    backend = backend or select_backend("inference")
    if not backend.can_pack():
        raise ValueError(
            f"The {backend.name} backend cannot multiply packed weights, so "
            f"an expert index would have nothing to be in the layout of."
        )

    conn = sqlite3.connect(db_path)
    # Reading every expert to quantize it, so the map is worth having -- but
    # sized against the machine rather than at a constant 32 GB, which on a
    # mixture-of-experts model large enough to want an index was the case
    # mmap_bytes measured at 3.7x slower.
    conn.execute(f"PRAGMA mmap_size = {mmap_bytes(db_path)}")
    conn.executescript(SCHEMA)
    conn.execute("DELETE FROM expert_index")
    conn.execute("DELETE FROM expert_index_layout")

    names = expert_tensor_names(conn)
    if not names:
        conn.close()
        raise ValueError(
            f"{db_path} holds no stacked expert tensors, so there is nothing "
            f"to index. This is for mixture-of-experts models."
        )

    started = time.time()
    next_id = 1
    written = 0
    total_bytes = 0

    for position, name in enumerate(names):
        rowid, dtype, shape, n_bytes = conn.execute(
            "SELECT id, dtype, shape, n_bytes FROM tensors WHERE name = ?",
            (name,),
        ).fetchone()
        dims = tuple(ast.literal_eval(shape))[::-1]
        n_experts, rows, cols = dims
        # The requested group size is a ceiling rather than a demand: it has
        # to divide the row, and gpt-oss's 2880 takes 64 where a 4096-wide
        # model would take 128.
        group = best_group(cols, group_size)
        stride = n_bytes // n_experts
        handle = conn.blobopen("tensors", "data", rowid, readonly=True)

        layout = None
        for expert in range(n_experts):
            raw = handle[expert * stride:(expert + 1) * stride]

            native = getattr(backend, f"dequantize_{dtype.lower()}", None)
            if native is not None:
                arr = native(raw, (rows, cols))
            else:
                arr = backend.from_numpy(
                    to_float32_any(raw, dtype).reshape(rows, cols))
            packed = backend.pack(arr, bits, group, compact=True)
            block = _to_bytes(backend, packed)

            if layout is None:
                q_bytes = _nbytes(backend, packed.q)
                s_bytes = _nbytes(backend, packed.scales)
                layout = Layout(name, next_id, n_experts, rows, cols, bits,
                                group, q_bytes, s_bytes)

            conn.execute("INSERT INTO expert_index (id, block) VALUES (?, ?)",
                         (next_id, block))
            next_id += 1
            written += 1
            total_bytes += len(block)

        handle.close()
        conn.execute(
            "INSERT INTO expert_index_layout (tensor, base_id, n_experts, "
            "rows, cols, bits, group_size, q_bytes, s_bytes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (layout.tensor, layout.base_id, layout.n_experts, layout.rows,
             layout.cols, layout.bits, layout.group_size, layout.q_bytes,
             layout.s_bytes),
        )
        conn.commit()
        if progress:
            progress(position + 1, len(names), name, written, total_bytes,
                     time.time() - started)

    conn.close()
    return {
        "tensors": len(names),
        "experts": written,
        "bytes": total_bytes,
        "bits": bits,
        "group_size": group_size,
        "seconds": time.time() - started,
    }


def drop(db_path: str) -> bool:
    """Remove the index. Returns whether there was one."""
    return blockindex.drop(db_path, TABLE)


_nbytes = blockindex.nbytes
_to_bytes = blockindex.to_bytes
