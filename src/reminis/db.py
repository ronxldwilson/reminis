"""How reminis opens a SQLite file, in one place.

Every database reminis touches is opened for one of four reasons, and the
pragmas that suit one are wrong for the others. Those settings used to be
written out by hand at each call site -- thirty connections, twenty-eight
loose ``PRAGMA`` lines -- which meant a measurement made against one path did
not reach the others. Delta packs kept a rollback journal for two releases
after the converter stopped using one, and the pack written by ``lora``, the
snapshot written by ``track`` and the export written by ``registry`` all kept
theirs longer still, for no reason beyond nobody having edited those lines.

The numbers quoted below were measured on a 2.5 GB model; see CHANGELOG.md
for the releases they come from.
"""

import sqlite3

# A file that did not exist a moment ago has nothing to roll back to, so the
# journal protects nothing and costs a second write of every byte. 64 KB pages
# were chosen by measuring reads as well as writes, since page size is stamped
# into the header and outlives the connection: against SQLite's 4 KB default
# they give 5.5x the write throughput, 3.6x sequential read, and 2.8x on the
# small random byte-range reads the expert index does.
BULK_WRITE_PRAGMAS = (
    "PRAGMA page_size=65536",
    "PRAGMA journal_mode=OFF",
    "PRAGMA synchronous=OFF",
    "PRAGMA cache_size=-256000",
)

# Same reasoning about the journal, but page_size is absent and the cache is
# left alone. Both would be wrong here: page_size cannot be changed on a
# populated file (SQLite ignores it silently), and raising the cache on this
# path measured *slower* than leaving it -- the pages being written are ones
# the copy just brought through, so a larger cache buys nothing and costs
# eviction work.
BULK_COPY_PRAGMAS = (
    "PRAGMA journal_mode=OFF",
    "PRAGMA synchronous=OFF",
)

# For files that are appended to over time and must survive a crash: a
# registry accumulating models, a training log written from inside a loop.
# WAL lets a reader in while a write is in flight, and NORMAL drops the
# fsync-per-commit that would otherwise land in the caller's step time.
APPEND_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA cache_size=-64000",
)


def _apply(conn: sqlite3.Connection, pragmas) -> sqlite3.Connection:
    for pragma in pragmas:
        conn.execute(pragma)
    return conn


def open_for_bulk_write(db_path: str) -> sqlite3.Connection:
    """Write a whole model into a file that does not exist yet.

    ``isolation_level=None`` hands transaction control to the caller, so the
    load runs inside one explicit BEGIN/COMMIT rather than sqlite3's implicit
    per-statement one.

    The journal is off, so a process killed mid-write leaves an unusable file
    rather than a rolled-back one. That is the right trade for every caller
    here: each unlinks its target first and writes it whole, so a crash means
    the output was never produced, which is what the caller would conclude
    from a rolled-back empty file anyway.
    """
    return _apply(sqlite3.connect(db_path, isolation_level=None), BULK_WRITE_PRAGMAS)


def open_for_bulk_copy(db_path: str) -> sqlite3.Connection:
    """Rewrite parts of a file that was copied from another one moments ago.

    ``apply`` and ``merge`` both start by copying a source database and then
    overwriting rows in the copy. The original is untouched on disk and the
    copy is deleted and rebuilt if the operation fails, so a journal protects
    a file nobody would keep.
    """
    return _apply(sqlite3.connect(db_path, isolation_level=None), BULK_COPY_PRAGMAS)


def open_for_append(db_path: str, foreign_keys: bool = False) -> sqlite3.Connection:
    """Open a long-lived database that is added to over time.

    ``foreign_keys`` is off by default because SQLite's own default is off and
    turning it on changes what writes are legal; only the registry, whose
    schema declares references, asks for it.
    """
    conn = _apply(sqlite3.connect(db_path), APPEND_PRAGMAS)
    if foreign_keys:
        conn.execute("PRAGMA foreign_keys=ON")
    return conn


def open_read_only(db_path: str) -> sqlite3.Connection:
    """Open a database that must not be modified.

    ``mode=ro`` is enforced by SQLite rather than by convention, so a stray
    write raises instead of quietly changing a model someone is reading.
    Unlike a plain connect, this raises if the file does not exist rather than
    creating an empty one.
    """
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


# ── what a tensor is ────────────────────────────────────────────────────────
#
# The columns below were declared in two schemas and written by eight separate
# INSERT statements, each spelling out its own placeholder list. That is the
# shape of mistake that shipped in delta packs for several releases: an insert
# naming nine columns and binding eight, which nothing caught because the only
# path reaching it was adding a tensor rather than changing one.
#
# Both the DDL and the statements are now built from this one tuple, so a
# placeholder list cannot disagree with a column list -- there is nothing left
# to keep in sync by hand.
TENSOR_COLUMNS = (
    "name",
    "shape",
    "dtype",
    "dtype_id",
    "n_elements",
    "n_bytes",
    "data",
)

# The column definitions, shared by the single-model schema and the registry's.
# Uniqueness is deliberately not declared here: a single-model file is unique
# on `name` alone, a registry on `(model_id, name)`, and putting the constraint
# in the shared text would make one of them wrong.
TENSOR_COLUMN_DDL = """    name       TEXT NOT NULL,
    shape      TEXT NOT NULL,
    dtype      TEXT NOT NULL,
    dtype_id   INTEGER NOT NULL,
    n_elements INTEGER NOT NULL,
    n_bytes    INTEGER NOT NULL,
    data       BLOB NOT NULL"""


def insert_sql(table: str, columns, verb: str = "INSERT") -> str:
    """An INSERT naming `columns`, with a placeholder for each.

    Generated rather than written out so the two lists cannot drift apart.
    `verb` carries the conflict clause -- "INSERT OR REPLACE" and so on.
    """
    return (
        f"{verb} INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' * len(columns))})"
    )


INSERT_TENSOR = insert_sql("tensors", TENSOR_COLUMNS)
INSERT_OR_REPLACE_TENSOR = insert_sql("tensors", TENSOR_COLUMNS, "INSERT OR REPLACE")

# The registry stores many models in one file, so every row carries the model
# it belongs to.
INSERT_REGISTRY_TENSOR = insert_sql("tensors", ("model_id",) + TENSOR_COLUMNS)

# `apply` rewrites tensors in a copy of the base model, so a name it inserts
# may or may not already be there. Built from the same column list, so the
# updated set cannot fall behind the inserted one either.
UPSERT_TENSOR = INSERT_TENSOR + (
    " ON CONFLICT(name) DO UPDATE SET "
    + ", ".join(f"{c}=excluded.{c}" for c in TENSOR_COLUMNS if c != "name")
)
