"""Many models in one database, with derived models stored as deltas.

Until now every artifact was its own file: a base model, a fine-tune, the pack
between them, an adapter, each a separate ``.db``. Keeping a base and three
fine-tunes meant four full copies of the weights.

A registry holds them together. Base models store their weights outright;
anything derived from one stores *only what differs*, using the same verified
delta encodings as :mod:`reminis.diff`. A base plus three LoRA fine-tunes costs
roughly the base plus three small deltas, not four bases.

    reminis registry add models.db  llama-1b.gguf        --name llama-1b
    reminis registry add models.db  ./sql-finetune/      --name llama-1b-sql   --parent llama-1b
    reminis registry add-lora models.db ./chat-adapter/  --name llama-1b-chat  --parent llama-1b
    reminis registry ls models.db

Models form a tree: a fine-tune of a fine-tune is stored against its immediate
parent, and resolving a tensor walks the chain from the root. Every model
records the hash of its own weights when it was added, so materialising one
later is checked rather than trusted.

Single-model databases are unchanged. A registry is a separate format for
holding a collection; ``convert`` and ``export`` still produce and consume
ordinary one-model files, and a registry can ingest or emit them.
"""

import json
import shutil
import sqlite3
import tempfile
import time

from reminis.db import (
    INSERT_REGISTRY_TENSOR,
    INSERT_TENSOR,
    SELECT_TENSOR_ROWS,
    TENSOR_COLUMN_DDL,
    open_for_append,
    open_for_bulk_write,
)
from pathlib import Path

REGISTRY_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS registry_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT UNIQUE NOT NULL,
    parent_id     INTEGER REFERENCES models(id),
    kind          TEXT NOT NULL,
    source_format TEXT,
    dtype_system  TEXT,
    weights_hash  TEXT NOT NULL,
    fingerprint   TEXT,
    n_tensors     INTEGER NOT NULL,
    logical_bytes INTEGER NOT NULL,
    stored_bytes  INTEGER NOT NULL,
    lossy         INTEGER NOT NULL DEFAULT 0,
    max_rel_error REAL,
    notes         TEXT,
    created_at    REAL NOT NULL
);

-- Weights of base models, stored outright. Same columns as a single-model
-- database, plus the model they belong to -- many models share this file, so
-- a tensor is identified by (model_id, name) rather than by name.
CREATE TABLE IF NOT EXISTS tensors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id   INTEGER NOT NULL REFERENCES models(id),
{TENSOR_COLUMN_DDL},
    UNIQUE (model_id, name)
);

-- What a derived model changes about its parent. Same encodings as a pack.
CREATE TABLE IF NOT EXISTS deltas (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id     INTEGER NOT NULL REFERENCES models(id),
    tensor_name  TEXT NOT NULL,
    shape        TEXT NOT NULL,
    dtype        TEXT NOT NULL,
    dtype_id     INTEGER NOT NULL,
    n_elements   INTEGER NOT NULL,
    encoding     TEXT NOT NULL,
    raw_bytes    INTEGER NOT NULL,
    stored_bytes INTEGER NOT NULL,
    rank         INTEGER,
    rel_error    REAL,
    data         BLOB NOT NULL,
    UNIQUE (model_id, tensor_name)
);

-- Tensors a derived model drops relative to its parent.
CREATE TABLE IF NOT EXISTS removed_tensors (
    model_id    INTEGER NOT NULL REFERENCES models(id),
    tensor_name TEXT NOT NULL,
    PRIMARY KEY (model_id, tensor_name)
);

CREATE TABLE IF NOT EXISTS model_meta (
    model_id INTEGER NOT NULL REFERENCES models(id),
    key      TEXT NOT NULL,
    value    TEXT NOT NULL,
    type     TEXT NOT NULL,
    PRIMARY KEY (model_id, key)
);

CREATE INDEX IF NOT EXISTS idx_reg_tensors ON tensors(model_id, name);
CREATE INDEX IF NOT EXISTS idx_reg_deltas  ON deltas(model_id, tensor_name);
"""

REGISTRY_FORMAT_VERSION = "1"

# A derived model resolves through its parents one tensor at a time. Chains
# this long are already pathological, and the limit turns a cycle introduced by
# a corrupt file into an error rather than a hang.
MAX_CHAIN_DEPTH = 64


class Registry:
    """A database holding several related models."""

    def __init__(self, path: str, create: bool = True):
        self.path = str(path)
        exists = Path(self.path).exists()
        if not exists and not create:
            raise FileNotFoundError(f"Registry not found: {self.path}")

        self.conn = open_for_append(self.path, foreign_keys=True)

        if exists:
            self._check_is_registry()
        self.conn.executescript(REGISTRY_SCHEMA)
        self.conn.execute(
            "INSERT OR IGNORE INTO registry_meta (key, value) VALUES "
            "('registry_format_version', ?)",
            (REGISTRY_FORMAT_VERSION,),
        )
        self.conn.commit()

    def _check_is_registry(self):
        """Refuse to graft registry tables onto a single-model database.

        Both formats have a `tensors` table with different columns, so opening
        one as the other would half-work and then produce nonsense. A
        single-model file has tensors but no models table.
        """
        names = {
            r[0]
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "tensors" in names and "models" not in names:
            self.conn.close()
            raise ValueError(
                f"{self.path} is a single-model database, not a registry.\n"
                "Create a registry at a new path and add this model to it:\n"
                f"  reminis registry add <registry.db> {self.path} --name <name>"
            )
        if "deltas" in names and "models" not in names:
            self.conn.close()
            raise ValueError(
                f"{self.path} is a delta pack, not a registry. Packs are applied "
                "with `reminis apply`."
            )

    # -- lookup ------------------------------------------------------------

    def _row(self, name: str):
        row = self.conn.execute(
            "SELECT id, name, parent_id, kind, weights_hash, n_tensors, "
            "logical_bytes, stored_bytes, lossy FROM models WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            known = [r[0] for r in self.conn.execute("SELECT name FROM models ORDER BY name")]
            raise KeyError(
                f"No model named '{name}' in {self.path}. "
                f"Known models: {', '.join(known) if known else 'none'}"
            )
        return row

    def __contains__(self, name: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM models WHERE name = ?", (name,)
            ).fetchone()
            is not None
        )

    def list_models(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT m.id, m.name, p.name, m.kind, m.n_tensors, m.logical_bytes, "
            "m.stored_bytes, m.lossy, m.max_rel_error, m.created_at, m.weights_hash "
            "FROM models m LEFT JOIN models p ON m.parent_id = p.id ORDER BY m.id"
        ).fetchall()
        return [
            {
                "id": r[0], "name": r[1], "parent": r[2], "kind": r[3],
                "n_tensors": r[4], "logical_bytes": r[5], "stored_bytes": r[6],
                "lossy": bool(r[7]), "max_rel_error": r[8], "created_at": r[9],
                "weights_hash": r[10],
            }
            for r in rows
        ]

    def _chain(self, model_id: int) -> list[int]:
        """Model ids from the root down to this one."""
        chain = []
        current = model_id
        seen = set()
        while current is not None:
            if current in seen or len(chain) > MAX_CHAIN_DEPTH:
                raise ValueError(
                    f"Model chain for id {model_id} is cyclic or deeper than "
                    f"{MAX_CHAIN_DEPTH}; the registry is corrupt."
                )
            seen.add(current)
            chain.append(current)
            row = self.conn.execute(
                "SELECT parent_id FROM models WHERE id = ?", (current,)
            ).fetchone()
            current = row[0] if row else None
        return list(reversed(chain))

    def _tensor_index(self, model_id: int) -> dict:
        """{tensor name: (shape, dtype, dtype_id, n_elements)} for a model.

        Built by walking the chain and applying each level's additions and
        removals, so a derived model reports the tensor set it actually has
        rather than its parent's.
        """
        index: dict = {}
        for level, mid in enumerate(self._chain(model_id)):
            if level == 0:
                for name, shape, dtype, dtype_id, n_elements in self.conn.execute(
                    "SELECT name, shape, dtype, dtype_id, n_elements FROM tensors "
                    "WHERE model_id = ?", (mid,)
                ):
                    index[name] = (shape, dtype, dtype_id, n_elements)
                continue

            for name, shape, dtype, dtype_id, n_elements in self.conn.execute(
                "SELECT tensor_name, shape, dtype, dtype_id, n_elements FROM deltas "
                "WHERE model_id = ?", (mid,)
            ):
                index[name] = (shape, dtype, dtype_id, n_elements)

            for (name,) in self.conn.execute(
                "SELECT tensor_name FROM removed_tensors WHERE model_id = ?", (mid,)
            ):
                index.pop(name, None)
        return index

    def _blob(self, chain: list[int], name: str) -> bytes | None:
        """Resolve one tensor's bytes by walking the chain from the root."""
        from reminis.diff import _decode_tensor

        row = self.conn.execute(
            "SELECT data FROM tensors WHERE model_id = ? AND name = ?",
            (chain[0], name),
        ).fetchone()
        blob = row[0] if row else None

        for mid in chain[1:]:
            dropped = self.conn.execute(
                "SELECT 1 FROM removed_tensors WHERE model_id = ? AND tensor_name = ?",
                (mid, name),
            ).fetchone()
            if dropped:
                blob = None
                continue
            entry = self.conn.execute(
                "SELECT encoding, dtype, data FROM deltas "
                "WHERE model_id = ? AND tensor_name = ?", (mid, name)
            ).fetchone()
            if entry:
                encoding, dtype, payload = entry
                blob = _decode_tensor(encoding, payload, blob, dtype, name)
        return blob

    def tensors(self, name: str):
        """Iterate a model's resolved tensors, one at a time.

        Yields ``(name, shape, dtype, dtype_id, n_elements, blob)`` in name
        order. Only one tensor is held at a time, so this works on models far
        larger than memory.
        """
        model_id = self._row(name)[0]
        chain = self._chain(model_id)
        index = self._tensor_index(model_id)
        for tensor_name in sorted(index):
            shape, dtype, dtype_id, n_elements = index[tensor_name]
            blob = self._blob(chain, tensor_name)
            if blob is None:
                raise ValueError(
                    f"Tensor '{tensor_name}' of model '{name}' could not be "
                    "resolved; the registry is inconsistent."
                )
            yield tensor_name, shape, dtype, dtype_id, n_elements, blob

    def weights_hash(self, name: str) -> str:
        """Hash a model's resolved weights, matching diff's ordering."""
        import hashlib

        h = hashlib.sha256()
        for _, _, _, _, _, blob in self.tensors(name):
            h.update(blob)
        return h.hexdigest()

    def meta(self, name: str) -> dict:
        model_id = self._row(name)[0]
        return {
            k: v
            for k, v in self.conn.execute(
                "SELECT key, value FROM model_meta WHERE model_id = ?", (model_id,)
            )
        }

    # -- adding ------------------------------------------------------------

    def add_base(
        self,
        source: str,
        name: str,
        notes: str | None = None,
        verbose: bool = True,
    ) -> dict:
        """Add a model in full. Accepts any format, or an existing .db."""
        if name in self:
            raise ValueError(f"A model named '{name}' is already in this registry")

        with _as_single_model_db(source, verbose=verbose) as db_path:
            src = sqlite3.connect(str(db_path))
            meta = {k: (v, t) for k, v, t in src.execute(
                "SELECT key, value, type FROM model_meta"
            )}

            model_id = self._insert_model(
                name=name, parent_id=None, kind="base",
                source_format=meta.get("reminis.source_format", ("gguf", ""))[0],
                dtype_system=meta.get("reminis.dtype_system", ("gguf", ""))[0],
                weights_hash="", fingerprint="", n_tensors=0,
                logical_bytes=0, stored_bytes=0, notes=notes,
            )

            n_tensors = 0
            total = 0
            for row in src.execute(SELECT_TENSOR_ROWS):
                self.conn.execute(
                    INSERT_REGISTRY_TENSOR,
                    (model_id, *row),
                )
                n_tensors += 1
                total += row[5]

            for key, (value, type_name) in meta.items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO model_meta (model_id, key, value, type) "
                    "VALUES (?,?,?,?)", (model_id, key, value, type_name),
                )

            from reminis.diff import _model_fingerprint, _weights_hash

            weights_hash = _weights_hash(src)
            fingerprint = _model_fingerprint(src)
            src.close()

            self.conn.execute(
                "UPDATE models SET weights_hash=?, fingerprint=?, n_tensors=?, "
                "logical_bytes=?, stored_bytes=? WHERE id=?",
                (weights_hash, fingerprint, n_tensors, total, total, model_id),
            )
            self.conn.commit()

        info = {
            "name": name, "kind": "base", "parent": None,
            "n_tensors": n_tensors, "logical_bytes": total, "stored_bytes": total,
            "weights_hash": weights_hash,
        }
        if verbose:
            _print_added(info)
        return info

    def add_derived(
        self,
        source: str,
        name: str,
        parent: str,
        lossy_tolerance: float | None = None,
        notes: str | None = None,
        verbose: bool = True,
    ) -> dict:
        """Add a model as the difference from an existing one.

        The source is a complete model; what gets stored is only what it
        changes about ``parent``. Uses the same encoder as ``reminis diff``, so
        a fine-tune costs what its pack would.
        """
        if name in self:
            raise ValueError(f"A model named '{name}' is already in this registry")
        parent_row = self._row(parent)

        from reminis.diff import diff_models

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            parent_db = tmp / "parent.db"
            self.materialize(parent, str(parent_db), verify=False, verbose=False)

            with _as_single_model_db(source, verbose=verbose) as source_db:
                pack = tmp / "pack.db"
                if verbose:
                    print(f"Diffing against '{parent}' ...")
                diff_models(
                    str(parent_db), str(source_db), str(pack),
                    verbose=False, lossy_tolerance=lossy_tolerance,
                )
                return self._ingest_pack(
                    str(pack), name, parent_row, kind="derived",
                    notes=notes, verbose=verbose,
                )

    def add_lora(
        self,
        adapter: str,
        name: str,
        parent: str,
        notes: str | None = None,
        verbose: bool = True,
    ) -> dict:
        """Add a peft LoRA adapter as a derived model.

        Stores the adapter's own factors, so the fine-tune costs a few MB and
        nothing is approximated.
        """
        if name in self:
            raise ValueError(f"A model named '{name}' is already in this registry")
        parent_row = self._row(parent)

        from reminis.lora import lora_to_delta_pack

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            parent_db = tmp / "parent.db"
            self.materialize(parent, str(parent_db), verify=False, verbose=False)
            pack = tmp / "pack.db"
            lora_to_delta_pack(adapter, str(parent_db), str(pack), verbose=False)
            return self._ingest_pack(
                str(pack), name, parent_row, kind="lora",
                notes=notes or f"peft adapter: {Path(adapter).name}", verbose=verbose,
            )

    def add_pack(
        self,
        pack: str,
        name: str,
        parent: str,
        notes: str | None = None,
        verbose: bool = True,
    ) -> dict:
        """Add a model from an existing delta pack against a registered parent."""
        if name in self:
            raise ValueError(f"A model named '{name}' is already in this registry")
        return self._ingest_pack(
            pack, name, self._row(parent), kind="derived", notes=notes, verbose=verbose
        )

    def _ingest_pack(self, pack_path, name, parent_row, kind, notes, verbose) -> dict:
        """Copy a delta pack's rows in as a derived model, then verify."""
        parent_id, parent_name = parent_row[0], parent_row[1]
        pack = sqlite3.connect(str(pack_path))
        meta = dict(pack.execute("SELECT key, value FROM delta_meta"))

        expected_base = meta.get("base_weights_hash")
        actual_base = parent_row[4]
        if expected_base and expected_base != actual_base:
            pack.close()
            raise ValueError(
                f"This pack was built against a different model than '{parent_name}'.\n"
                f"  pack expects base hash: {expected_base[:16]}...\n"
                f"  '{parent_name}' has:      {actual_base[:16]}...\n"
                "Storing it would produce a model that cannot be reconstructed."
            )

        lossy = meta.get("lossy") == "true"
        model_id = self._insert_model(
            name=name, parent_id=parent_id, kind=kind,
            source_format=None, dtype_system=None,
            weights_hash=meta.get("reconstructed_weights_hash")
            or meta.get("target_weights_hash", ""),
            fingerprint=meta.get("target_fingerprint", ""),
            n_tensors=0, logical_bytes=0, stored_bytes=0,
            lossy=lossy, max_rel_error=float(meta.get("max_rel_error", "0") or 0),
            notes=notes,
        )

        stored = 0
        for row in pack.execute(
            "SELECT tensor_name, shape, dtype, dtype_id, n_elements, encoding, "
            "raw_bytes, stored_bytes, rank, rel_error, data FROM deltas"
        ):
            self.conn.execute(
                "INSERT INTO deltas (model_id, tensor_name, shape, dtype, dtype_id, "
                "n_elements, encoding, raw_bytes, stored_bytes, rank, rel_error, data) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (model_id, *row),
            )
            stored += row[7]

        for tensor_name in json.loads(meta.get("tensors_removed", "[]")):
            self.conn.execute(
                "INSERT OR IGNORE INTO removed_tensors (model_id, tensor_name) "
                "VALUES (?,?)", (model_id, tensor_name),
            )

        # Carry the parent's metadata forward; a fine-tune of llama is still
        # llama, and `info` on it should say so.
        self.conn.execute(
            "INSERT OR REPLACE INTO model_meta (model_id, key, value, type) "
            "SELECT ?, key, value, type FROM model_meta WHERE model_id = ?",
            (model_id, parent_id),
        )
        pack.close()
        self.conn.commit()

        index = self._tensor_index(model_id)
        logical = 0
        for _, _, _, _, _, blob in self.tensors(name):
            logical += len(blob)

        self.conn.execute(
            "UPDATE models SET n_tensors=?, logical_bytes=?, stored_bytes=? WHERE id=?",
            (len(index), logical, stored, model_id),
        )
        self.conn.commit()

        # The recorded hash is a promise that this model can be rebuilt. Check
        # it now, while the source is still around to fix, rather than the
        # first time someone tries to export it.
        recorded = self.conn.execute(
            "SELECT weights_hash FROM models WHERE id = ?", (model_id,)
        ).fetchone()[0]
        if recorded:
            actual = self.weights_hash(name)
            if actual != recorded:
                self.conn.execute("DELETE FROM deltas WHERE model_id=?", (model_id,))
                self.conn.execute("DELETE FROM removed_tensors WHERE model_id=?", (model_id,))
                self.conn.execute("DELETE FROM model_meta WHERE model_id=?", (model_id,))
                self.conn.execute("DELETE FROM models WHERE id=?", (model_id,))
                self.conn.commit()
                raise ValueError(
                    f"Storing '{name}' did not reproduce the expected weights.\n"
                    f"  expected: {recorded[:16]}...\n  actual:   {actual[:16]}...\n"
                    "Nothing was added."
                )

        info = {
            "name": name, "kind": kind, "parent": parent_name,
            "n_tensors": len(index), "logical_bytes": logical,
            "stored_bytes": stored, "weights_hash": recorded, "lossy": lossy,
        }
        if verbose:
            _print_added(info)
        return info

    def _insert_model(self, **kw) -> int:
        params = {
            "source_format": None, "dtype_system": None, "fingerprint": "",
            "lossy": 0, "max_rel_error": None, "notes": None,
            "created_at": time.time(),
            **kw,
        }
        params["lossy"] = int(bool(params["lossy"]))
        cur = self.conn.execute(
            "INSERT INTO models (name, parent_id, kind, source_format, dtype_system, "
            "weights_hash, fingerprint, n_tensors, logical_bytes, stored_bytes, "
            "lossy, max_rel_error, notes, created_at) "
            "VALUES (:name,:parent_id,:kind,:source_format,:dtype_system,:weights_hash,"
            ":fingerprint,:n_tensors,:logical_bytes,:stored_bytes,:lossy,:max_rel_error,"
            ":notes,:created_at)",
            params,
        )
        return cur.lastrowid

    # -- getting models back out -------------------------------------------

    def materialize(
        self,
        name: str,
        output_db: str,
        verify: bool = True,
        verbose: bool = True,
    ) -> str:
        """Write one model out as an ordinary single-model database."""
        from reminis.converter import SCHEMA

        t0 = time.time()
        row = self._row(name)
        Path(output_db).unlink(missing_ok=True)
        out = open_for_bulk_write(output_db)
        out.executescript(SCHEMA)
        out.execute("BEGIN")

        import hashlib

        h = hashlib.sha256()
        n = 0
        for tensor_name, shape, dtype, dtype_id, n_elements, blob in self.tensors(name):
            out.execute(
                INSERT_TENSOR,
                (tensor_name, shape, dtype, dtype_id, n_elements, len(blob), blob),
            )
            h.update(blob)
            n += 1

        for key, value, type_name in self.conn.execute(
            "SELECT key, value, type FROM model_meta WHERE model_id = ?", (row[0],)
        ):
            out.execute(
                "INSERT OR REPLACE INTO model_meta (key, value, type) VALUES (?,?,?)",
                (key, value, type_name),
            )
        out.commit()
        out.close()

        if verify and row[4]:
            actual = h.hexdigest()
            if actual != row[4]:
                Path(output_db).unlink(missing_ok=True)
                raise ValueError(
                    f"Materialising '{name}' produced different weights than recorded.\n"
                    f"  expected: {row[4][:16]}...\n  actual:   {actual[:16]}..."
                )

        if verbose:
            print(f"Wrote '{name}' ({n} tensors) to {output_db}")
            if verify and row[4]:
                print("Verified against the hash recorded when it was added.")
            print(f"Took {time.time() - t0:.1f}s")
        return output_db

    def export(self, name: str, output: str, fmt: str = "auto", verbose: bool = True) -> str:
        """Materialise a model and write it out as GGUF or safetensors."""
        from reminis.converter import sqlite_to_gguf
        from reminis.safetensors_io import sqlite_to_safetensors

        if fmt == "auto":
            suffix = Path(output).suffix
            fmt = "safetensors" if suffix == ".safetensors" else (
                "gguf" if suffix == ".gguf" else
                (self.meta(name).get("reminis.source_format") or "gguf")
            )

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "model.db")
            self.materialize(name, db, verify=True, verbose=False)
            if fmt == "safetensors":
                return sqlite_to_safetensors(db, output, verbose=verbose)
            return sqlite_to_gguf(db, output, verbose=verbose)

    # -- housekeeping ------------------------------------------------------

    def children(self, name: str) -> list[str]:
        model_id = self._row(name)[0]
        return [
            r[0]
            for r in self.conn.execute(
                "SELECT name FROM models WHERE parent_id = ? ORDER BY name", (model_id,)
            )
        ]

    def remove(self, name: str, verbose: bool = True) -> None:
        """Remove a model. Refuses if anything is derived from it."""
        row = self._row(name)
        kids = self.children(name)
        if kids:
            raise ValueError(
                f"'{name}' is the parent of: {', '.join(kids)}.\n"
                "Removing it would leave them unresolvable. Remove those first, "
                "or materialise them into base models."
            )
        model_id = row[0]
        for table in ("tensors", "deltas", "removed_tensors", "model_meta"):
            self.conn.execute(f"DELETE FROM {table} WHERE model_id = ?", (model_id,))
        self.conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        self.conn.commit()
        # SQLite keeps freed pages for reuse; reclaim them so removing a large
        # model actually shrinks the file.
        self.conn.execute("VACUUM")
        if verbose:
            print(f"Removed '{name}' from {self.path}")

    def stats(self) -> dict:
        models = self.list_models()
        logical = sum(m["logical_bytes"] for m in models)
        file_bytes = Path(self.path).stat().st_size
        return {
            "path": self.path,
            "models": len(models),
            "bases": sum(1 for m in models if m["kind"] == "base"),
            "derived": sum(1 for m in models if m["kind"] != "base"),
            "logical_bytes": logical,
            "file_bytes": file_bytes,
            "savings": (logical - file_bytes) / logical if logical else 0.0,
        }

    def close(self):
        self.conn.commit()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _as_single_model_db:
    """Context manager yielding a single-model .db path for any source.

    A ``.db`` is used where it is; a GGUF or safetensors model is converted
    into a temporary one that is cleaned up afterwards.
    """

    def __init__(self, source: str, verbose: bool = True):
        self.source = str(source)
        self.verbose = verbose
        self._tmp = None

    def __enter__(self) -> str:
        src = Path(self.source)
        if not src.exists():
            raise FileNotFoundError(f"Not found: {src}")

        if src.is_file() and src.suffix == ".db":
            return str(src)

        from reminis.converter import gguf_to_sqlite
        from reminis.safetensors_io import safetensors_to_sqlite

        self._tmp = tempfile.TemporaryDirectory()
        db = str(Path(self._tmp.name) / "imported.db")
        if src.is_dir() or src.suffix == ".safetensors" or src.name.endswith(".index.json"):
            safetensors_to_sqlite(self.source, db, verbose=False)
        else:
            gguf_to_sqlite(self.source, db, verbose=False)
        return db

    def __exit__(self, *exc):
        if self._tmp is not None:
            self._tmp.cleanup()


def _print_added(info: dict):
    from reminis.diff import _fmt_bytes

    parent = f" (derived from '{info['parent']}')" if info.get("parent") else ""
    print(f"\nAdded '{info['name']}'{parent}")
    print(f"  tensors: {info['n_tensors']}")
    print(f"  full model would be: {_fmt_bytes(info['logical_bytes'])}")
    print(f"  stored in registry:  {_fmt_bytes(info['stored_bytes'])}", end="")
    if info["logical_bytes"]:
        pct = info["stored_bytes"] / info["logical_bytes"] * 100
        print(f"  ({pct:.1f}%)")
    else:
        print()
    if info.get("lossy"):
        print("  NOTE: stored with lossy low-rank encoding")
