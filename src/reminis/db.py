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

import os
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


def mmap_bytes(db_path: str) -> int:
    """How much of the file to map, which is all of it or none of it.

    Mapping the file rather than copying each blob through sqlite's own
    buffer measured 4.1 GB/s to 6.7 GB/s on a 258 MB model, and it is free
    there because the whole thing fits in memory several times over.

    It stops being free when the file is larger than the machine. A mapped
    page that has been touched stays resident until the kernel decides
    otherwise, so on a 22.9 GB database with 16 GB of RAM the map competes
    for memory with the weights the model is trying to keep, and every read
    pays for it: measured 1.70 ms per expert block mapped against 0.46 ms
    unmapped, which is 3.7x the wrong way.

    So this maps the file when it comfortably fits and does not when it does
    not, rather than asking for a fixed number of gigabytes and hoping. Half
    of physical memory is the line, which is where the two measured points
    fall either side of: a 4.4 GB model on a 16 GB machine is 7% faster
    mapped, and a 22.9 GB one is 3.7x slower.

    This lived on WeightStore, and the three other places that map a file
    each named their own constant instead -- 8 GB here, 32 GB in the expert
    index, nothing at all in packed_index. The 32 GB is the number the
    paragraph above was written to argue against, on the mixture-of-experts
    models where an index is worth building in the first place.
    """
    try:
        size = os.path.getsize(db_path)
        available = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        return 0
    return size if size * 2 <= available else 0


def open_for_read(db_path: str, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open a model to read tensors out of, mapped if the file fits.

    ``query_only`` so a stray write raises rather than changing a model
    someone is reading, and the map sized by the policy above.

    This is ``open_read_only`` plus the map: ``mode=ro`` is enforced by
    SQLite rather than by convention, and it raises on a missing file
    instead of creating an empty one to fail against later. The callers
    unified here were split between the two -- the read-ahead pool used
    ``mode=ro``, the weight store a plain connect -- and ``mode=ro`` is the
    one that cannot be got wrong.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                           check_same_thread=check_same_thread)
    conn.execute("PRAGMA query_only = 1")
    conn.execute(f"PRAGMA mmap_size = {mmap_bytes(db_path)}")
    return conn


# How many bytes of read-ahead to hold. The things being read are whole
# tensors and one of them can be half a gigabyte, so this is a byte budget
# rather than a count -- a fixed queue depth would hold four embedding
# matrices on one model and four norms on another.
READ_AHEAD_BYTES = 512_000_000

# Writes to one file serialize; reads do not. Measured on a 4.4 GB model,
# reading every tensor by name: 3570 MB/s on one thread, 15079 on four,
# 16056 on eight. Four is where the curve flattens, and it leaves cores for
# whatever the caller is doing with the bytes.
READ_THREADS = 4


def read_blobs_ahead(db_path: str, names, threads: int = READ_THREADS,
                     budget: int = READ_AHEAD_BYTES):
    """Yield ``(name, blob)`` for `names`, **in order**, read on other threads.

    Every caller that walks a model does the same thing: read a tensor, do
    something with it, read the next one. The read blocks the work and the
    work blocks the read, so neither the disk nor the processor is ever busy.

    SQLite serializes writers and does not serialize readers, so the read
    half parallelises even though the write half cannot. Each worker gets its
    own connection -- a connection is not safe to share across threads, and
    `mmap_size` lets several of them touch the same pages without copying.

    Order is preserved because callers depend on it: an export has already
    written the offsets its tensors must land at, so reading them out of
    order would produce a file whose header lies about its contents.

    The read-ahead is bounded by bytes rather than by tensors, and always
    allows one, so a model whose largest tensor exceeds the budget still
    makes progress instead of deadlocking.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    import threading as _threading

    names = list(names)
    if not names:
        return

    local = _threading.local()

    def connection():
        conn = getattr(local, "conn", None)
        if conn is None:
            # Mapping the file rather than copying each blob through
            # SQLite's own buffer: 15079 MB/s against 11486 on four threads.
            conn = open_for_read(db_path, check_same_thread=False)
            local.conn = conn
        return conn

    def fetch(name):
        row = connection().execute(
            "SELECT data FROM tensors WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise KeyError(f"no tensor named '{name}' in {db_path}")
        return row[0]

    # n_bytes is a column, so the budget can be charged before a blob is read
    # rather than guessed at afterwards. One query, no weights.
    sizer = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        sizes = dict(sizer.execute("SELECT name, n_bytes FROM tensors"))
    finally:
        sizer.close()

    pool = ThreadPoolExecutor(max_workers=max(1, threads),
                              thread_name_prefix="reminis-read")
    try:
        pending = deque()
        in_flight = 0
        index = 0

        while index < len(names) or pending:
            # Read ahead while there is room, always allowing one so a tensor
            # larger than the whole budget still moves.
            while index < len(names) and (not pending or in_flight < budget):
                name = names[index]
                pending.append((name, pool.submit(fetch, name)))
                in_flight += sizes.get(name, 0)
                index += 1

            name, future = pending.popleft()
            blob = future.result()
            in_flight -= sizes.get(name, 0)
            yield name, blob
            del blob
    finally:
        pool.shutdown(wait=True)


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


def select_sql(table: str, columns, suffix: str = "") -> str:
    """A SELECT naming `columns`, in the order they are declared.

    The read-side counterpart of `insert_sql`, and it exists for the same
    reason. `registry.add_base` reads a row with one of these and binds it
    straight into an INSERT built from the same tuple -- so a column added
    to one list and not the other does not raise, it silently writes the
    values into the wrong columns.
    """
    return f"SELECT {', '.join(columns)} FROM {table}{suffix}"


# Every caller that walks a whole model wants the same seven columns in
# declaration order, oldest row first.
SELECT_TENSOR_ROWS = select_sql("tensors", TENSOR_COLUMNS, " ORDER BY id")

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
