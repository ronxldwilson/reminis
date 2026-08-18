"""A materialized index over a dense model's weights, already packed.

`expert_index` exists because a mixture of experts decides at query time
which eighth of its weights to read, and re-deriving the kernel's layout
per token was most of the cost. A dense model has the opposite problem and
the same answer: it reads *every* weight, once, at load -- and pays to
unpack and re-pack all of them before the first token appears.

That cost is not small and it is not amortised, because nothing is kept.
Measured on Qwen3.8-27B at 3 bits, unpacking 27.3 billion i-quantized
weights through numpy and re-quantizing them for the kernel took 466
seconds, every run, to produce 11.4 GB that the previous run had already
produced and discarded.

It is a pure function of bytes already in the database, so it belongs in
the database:

    reminis prepare model.db --weights          build it
    reminis prepare model.db --weights --drop   throw it away

With it, loading a weight is a primary-key seek and a memcpy into the
layout the quantized matmul already wants. Nothing is decoded.

What it costs is disk. The index is a second copy of the weights at the
chosen width -- roughly the size of the model -- and it is redundant,
derived, and droppable, which is the trade an index always offers. The
original tensors are untouched, so the database is still the model and
still converts back.

One table rather than one per tensor, and one row per tensor rather than
one per expert: a dense weight is read whole or not at all, so there is
nothing finer to seek to.
"""

import ast
import sqlite3
import time

from reminis import blockindex

# The suffixes worth storing packed: the per-layer matrices, which are
# nearly all of a model's bytes and are only ever the right-hand side of a
# matrix multiply. This mirrors `WeightStore._PACKABLE`, and the two must
# agree -- a tensor packed here that the store would not have packed would
# be read back into a shape the forward pass does not expect.
PACKABLE = (
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    "attn_qkv.weight", "attn_gate.weight", "ssm_out.weight",
)

# The vocabulary-sized matrices, which are not per-layer but are the two
# largest tensors in most models and the most expensive to unpack.
PACKABLE_EMBED = ("token_embd.weight", "output.weight")

DEFAULT_BITS = 4
DEFAULT_GROUP = 128

TABLE = "packed_index"
SCHEMA = blockindex.schema(TABLE, "    row_id     INTEGER NOT NULL,")


class Layout(blockindex.BlockLayout):
    """Where one tensor's packed form lives, and how to read it back.

    One row per tensor, so the identity is a single `row_id`.
    """

    __slots__ = ("row_id",)
    COLUMNS = ("tensor", "row_id") + blockindex.GEOMETRY_COLUMNS

    @property
    def shape(self) -> tuple:
        return (self.rows, self.cols)


def read_layouts(conn: sqlite3.Connection) -> dict:
    """Every packed tensor, or an empty mapping if there is no index."""
    return blockindex.read_layouts(conn, TABLE, Layout)


def packable(name: str) -> bool:
    if name in PACKABLE_EMBED:
        return True
    return name.startswith("blk.") and name.endswith(PACKABLE)


def tensor_names(conn: sqlite3.Connection) -> list:
    """The dense tensors worth packing, in the order a forward pass wants.

    Two dimensions only. A stacked expert tensor is three, and belongs to
    `expert_index`, which can seek within it.
    """
    out = []
    for name, shape in conn.execute("SELECT name, shape FROM tensors"):
        if not packable(name):
            continue
        if len(ast.literal_eval(shape)) != 2:
            continue
        out.append(name)

    def order(name):
        if name in PACKABLE_EMBED:
            return (-1, PACKABLE_EMBED.index(name))
        parts = name.split(".")
        suffix = ".".join(parts[2:])
        return (int(parts[1]),
                PACKABLE.index(suffix) if suffix in PACKABLE else 99)

    return sorted(out, key=order)


def build(db_path: str, backend=None, bits: int = DEFAULT_BITS,
          group_size: int = DEFAULT_GROUP, progress=None) -> dict:
    """Build the index, replacing any existing one."""
    from reminis.backend import best_group, select as select_backend
    from reminis.dtypes import is_float_dtype, to_float32_any

    backend = backend or select_backend("inference")
    if not backend.can_pack():
        raise ValueError(
            f"The {backend.name} backend cannot multiply packed weights, so "
            f"a packed index would have nothing to be in the layout of."
        )

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.execute("DELETE FROM packed_index")
    conn.execute("DELETE FROM packed_index_layout")

    names = tensor_names(conn)
    if not names:
        conn.close()
        raise ValueError(
            f"{db_path} holds no dense weight matrices to pack."
        )

    started = time.time()
    written = 0
    total_bytes = 0

    for position, name in enumerate(names, start=1):
        shape, dtype, blob = conn.execute(
            "SELECT shape, dtype, data FROM tensors WHERE name = ?", (name,)
        ).fetchone()
        dims = tuple(ast.literal_eval(shape))[::-1]
        rows, cols = dims
        group = best_group(cols, group_size)

        if is_float_dtype(dtype):
            arr = backend.from_bytes(blob, dtype, dims)
        else:
            arr = backend.from_numpy(to_float32_any(blob, dtype).reshape(dims))
        packed = backend.pack(arr, bits, group, compact=True)
        del arr
        block = _to_bytes(backend, packed)

        conn.execute("INSERT INTO packed_index (id, block) VALUES (?, ?)",
                     (position, block))
        conn.execute(
            "INSERT INTO packed_index_layout (tensor, row_id, rows, cols, "
            "bits, group_size, q_bytes, s_bytes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, position, rows, cols, bits, group,
             _nbytes(backend, packed.q), _nbytes(backend, packed.scales)),
        )
        written += 1
        total_bytes += len(block)
        del packed, block
        if position % 16 == 0:
            conn.commit()
        if progress:
            progress(position, len(names), name, total_bytes,
                     time.time() - started)

    conn.commit()
    conn.close()
    return {
        "tensors": written,
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
