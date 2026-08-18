"""What the two materialized indexes have in common.

`packed_index` and `expert_index` are the same idea applied to different
tensors: take weights the backend would otherwise unpack on every forward
pass, do that work once at build time, and store the result in the layout
the matmul kernel wants. One does it for the dense 2-D matrices, the other
for the stacked 3-D expert tensors, where a row is one expert rather than
one tensor.

They were written separately and stayed almost identical, which is how
they came to disagree. `expert_index.build` mapped 32 GB of the file and
`packed_index.build` mapped none; the two `drop` functions differed only
in a table name and a comment. What is shared here is everything that does
not depend on which of the two is being built: the geometry of a stored
block, the SQL around it, and the pair of backend helpers.

What is not shared is what a row *is*. A packed row is a whole tensor and
knows its `row_id`; an expert row is one expert of a stack and knows the
`base_id` its siblings count from. That difference is the reason there are
two modules, and it stays in them.
"""

import sqlite3

import numpy as np

# The block geometry, identical in both indexes: how many rows and columns
# the weight had, and how many bytes its codes and scales occupy.
GEOMETRY_COLUMNS = ("rows", "cols", "bits", "group_size", "q_bytes", "s_bytes")

GEOMETRY_DDL = """    rows       INTEGER NOT NULL,
    cols       INTEGER NOT NULL,
    bits       INTEGER NOT NULL,
    group_size INTEGER NOT NULL,
    q_bytes    INTEGER NOT NULL,
    s_bytes    INTEGER NOT NULL"""


class BlockLayout:
    """Where one stored block lives, and how to read it back.

    Subclasses declare `COLUMNS` -- the layout table's columns in order --
    and the extra `__slots__` those columns need beyond the geometry. The
    order matters twice over: it is the order the constructor takes its
    arguments in, and the order `read_layouts` selects them in, and they
    are the same tuple so they cannot come apart.
    """

    __slots__ = ("tensor",) + GEOMETRY_COLUMNS

    #: Overridden by each index. ("tensor",) + identity columns + geometry.
    COLUMNS: tuple = ()

    def __init__(self, *values):
        if len(values) != len(self.COLUMNS):
            raise TypeError(
                f"{type(self).__name__} takes {len(self.COLUMNS)} values "
                f"({', '.join(self.COLUMNS)}), got {len(values)}"
            )
        for name, value in zip(self.COLUMNS, values):
            setattr(self, name, value)

    @property
    def block_bytes(self) -> int:
        return self.q_bytes + 2 * self.s_bytes

    def split(self, raw):
        """Codes, scales and biases as numpy views onto the stored bytes.

        The three arrays a quantized matmul needs are stored end to end in
        one blob rather than in three columns, so reading a block is one
        seek rather than three. Nothing is copied here; the copy happens
        once, when the backend adopts them.
        """
        q_end = self.q_bytes
        s_end = q_end + self.s_bytes
        words = np.frombuffer(raw, dtype=np.uint32, count=self.q_bytes // 4)
        scales = np.frombuffer(raw, dtype=np.float16,
                               count=self.s_bytes // 2, offset=q_end)
        biases = np.frombuffer(raw, dtype=np.float16,
                               count=self.s_bytes // 2, offset=s_end)
        groups = self.cols // self.group_size
        return (words.reshape(self.rows, -1),
                scales.reshape(self.rows, groups),
                biases.reshape(self.rows, groups))


def schema(table: str, identity_ddl: str) -> str:
    """The two tables an index is: the blocks, and where each one lives.

    Built from `GEOMETRY_DDL` rather than spelled out per index, so the
    columns cannot drift from the ones `BlockLayout` reads.
    """
    return f"""
CREATE TABLE IF NOT EXISTS {table} (
    id    INTEGER PRIMARY KEY,
    block BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS {table}_layout (
    tensor     TEXT PRIMARY KEY,
{identity_ddl}
{GEOMETRY_DDL}
);
"""


def read_layouts(conn: sqlite3.Connection, table: str, cls) -> dict:
    """Every indexed tensor, or an empty mapping if there is no index.

    An absent index is ordinary rather than an error: it is derived data
    and `drop` exists, so every caller has to cope with it not being there.
    """
    try:
        rows = conn.execute(
            f"SELECT {', '.join(cls.COLUMNS)} FROM {table}_layout"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {row[0]: cls(*row) for row in rows}


def drop(db_path: str, table: str) -> bool:
    """Remove the index. Returns whether there was one."""
    conn = sqlite3.connect(db_path)
    try:
        existed = bool(conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' "
            "AND name=?", (table,)
        ).fetchone()[0])
        conn.executescript(
            f"DROP TABLE IF EXISTS {table};"
            f"DROP TABLE IF EXISTS {table}_layout;"
        )
        conn.commit()
        # The file keeps the freed pages otherwise, and an index that was
        # the same size as the model is not a rounding error.
        conn.execute("VACUUM")
        return existed
    finally:
        conn.close()


def nbytes(backend, array) -> int:
    return int(np.asarray(backend.to_host(array)).nbytes)


def to_bytes(backend, packed) -> bytes:
    """One quantized weight as the bytes the index stores."""
    parts = [backend.to_host(packed.q),
             backend.to_host(packed.scales),
             backend.to_host(packed.biases)]
    return b"".join(np.ascontiguousarray(p).tobytes() for p in parts)
