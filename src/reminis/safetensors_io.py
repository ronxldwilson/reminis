"""Read and write safetensors files, sharded or single.

Nobody fine-tunes in GGUF. Training happens in PyTorch/transformers/peft, which
emits safetensors, so reading that format directly removes the llama.cpp
conversion step between a fine-tune and reminis.

The format needs no library: an 8-byte little-endian header length, a JSON
header mapping tensor names to ``{dtype, shape, data_offsets}``, then raw bytes.
We parse it by hand rather than using ``safetensors.numpy``, whose dtype list
has no BF16 and no FP8 -- it would silently refuse the dtype most fine-tunes
are actually saved in.

Shape convention: reminis stores ``shape`` reversed relative to the data
layout, because that is what GGUF does and the diff, low-rank, and viewer code
all assume it. A safetensors tensor of logical shape ``[out, in]`` is therefore
stored as ``[in, out]``, and reversed back on export.
"""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from reminis.converter import human_bytes
from reminis.db import read_blobs_ahead
from reminis.dtypes import (
    DTYPE_SYSTEM_GGUF,
    DTYPE_SYSTEM_SAFETENSORS,
    GGML_TO_SAFETENSORS_DTYPE,
    SAFETENSORS_DTYPE_IDS,
    SAFETENSORS_DTYPES,
    dtype_system,
    from_float32,
    to_float32,
)

# A safetensors header is JSON describing tensor offsets; even for a very large
# sharded model it stays in the low megabytes. A wildly larger value means a
# corrupt or non-safetensors file, and reading it would allocate that much.
MAX_HEADER_BYTES = 100_000_000

INDEX_FILENAME = "model.safetensors.index.json"
DEFAULT_FILENAME = "model.safetensors"


def read_header(path: Path) -> tuple[dict, int]:
    """Parse a safetensors header. Returns (header, offset of the data block)."""
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        header_len = int.from_bytes(raw_len, "little")
        if header_len <= 0 or header_len > MAX_HEADER_BYTES:
            raise ValueError(
                f"{path} declares a {header_len}-byte header; this is not a "
                "safetensors file, or it is corrupt"
            )
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def _entries(header: dict) -> dict:
    """Tensor entries only, with the reserved metadata key removed."""
    return {k: v for k, v in header.items() if k != "__metadata__"}


def resolve_shards(path: Path) -> tuple[list[Path], dict]:
    """Work out which files make up a model, and its index metadata.

    Accepts a single ``.safetensors`` file, a ``*.index.json``, or a directory
    containing either. Large models ship split across
    ``model-0000N-of-0000M.safetensors`` with an index naming the shards.
    """
    if path.is_dir():
        index = path / INDEX_FILENAME
        if index.exists():
            path = index
        elif (path / DEFAULT_FILENAME).exists():
            path = path / DEFAULT_FILENAME
        else:
            shards = sorted(path.glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError(f"No safetensors files found in {path}")
            return shards, {}

    if path.suffix == ".json":
        index = json.loads(path.read_text())
        weight_map = index.get("weight_map")
        if not weight_map:
            raise ValueError(f"{path} has no weight_map; not a safetensors index")
        # dict.fromkeys keeps first-seen order, so shards stay in index order.
        names = list(dict.fromkeys(weight_map.values()))
        return [path.parent / n for n in names], index.get("metadata", {})

    return [path], {}


def _read_config(model_dir: Path) -> dict:
    """Load a sibling config.json, which is where the real architecture lives.

    A safetensors file carries only an optional ``__metadata__``, usually just
    ``{"format": "pt"}``. Without config.json, `reminis info` on a safetensors
    model would say almost nothing.
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _read_tokenizer(model_dir: Path) -> list[tuple[str, str, str]]:
    """A sibling tokenizer.json as the ``tokenizer.ggml.*`` rows reminis reads.

    A GGUF carries its tokenizer inside the file, so a converted text model
    can be run straight out of the database. safetensors carries none, and
    without this the database holds a model that computes correct logits and
    cannot say which words they stand for -- which is the same completeness
    gap that dropping array metadata on export turned out to be in 0.32.0.

    Only byte-level BPE is ingested. That is what the ``gpt2`` reader
    implements, and claiming a vocabulary reminis cannot drive would be worse
    than leaving it out: a wrong tokenizer decodes to fluent nonsense.
    """
    path = model_dir / "tokenizer.json"
    if not path.exists():
        return []
    try:
        spec = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    model = spec.get("model") or {}
    if str(model.get("type", "")).upper() != "BPE":
        return []
    vocab = model.get("vocab") or {}
    merges = model.get("merges") or []
    if not vocab or not merges:
        return []

    # `added_tokens` are the specials -- <|startoftranscript|>, the language
    # and task markers, <|endoftext|>, and Whisper's timestamps. They have to
    # be matched whole rather than split, which is what token type 3 means to
    # the reader. Their ids run *past* the end of `vocab`, so the table is
    # sized to cover both rather than to the vocabulary alone.
    added = {int(entry["id"]): entry["content"]
             for entry in (spec.get("added_tokens") or [])
             if "id" in entry and "content" in entry}

    size = max(max(vocab.values()), max(added, default=-1)) + 1
    tokens = [""] * size
    for token, index in vocab.items():
        tokens[index] = token
    for index, content in added.items():
        tokens[index] = content
    types = [3 if i in added else 1 for i in range(size)]

    # Older tokenizer.json files write each merge as "a b"; newer ones write
    # a two-element list. The reader wants the string form.
    flat = []
    for merge in merges:
        flat.append(merge if isinstance(merge, str) else " ".join(merge))

    # Stored as JSON rather than as a Python repr. The reader accepts both,
    # but `ast.literal_eval` on half a megabyte of vocabulary costs 82 ms
    # against `json.loads`'s 4.6 ms, and a transcription pays that three
    # times before it can name a single token.
    return [
        ("tokenizer.ggml.model", "gpt2", "string"),
        ("tokenizer.ggml.tokens", json.dumps(tokens), "json"),
        ("tokenizer.ggml.merges", json.dumps(flat), "json"),
        ("tokenizer.ggml.token_type", json.dumps(types), "json"),
    ]


def _tokenizer_meta_from_config(config: dict, arch) -> list[tuple[str, str, str]]:
    """The tokenizer settings a GGUF records and safetensors does not.

    Two of these decide whether a run is right rather than merely fast.
    The pre-tokenizer picks the splitter, and an unrecognised name falls
    back to GPT-2's, which disagrees with this model on every number. The
    end-of-sequence id is what lets generation stop; without it a reply
    runs to the token limit whatever the model intended.
    """
    rows = []
    if arch.pretokenizer:
        rows.append(("tokenizer.ggml.pre", arch.pretokenizer, "string"))

    gen = config.get("generation_config") or {}

    def first(value):
        """A model may name several end tokens; the reader takes one."""
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    bos = first(config.get("bos_token_id", gen.get("bos_token_id")))
    eos = first(config.get("eos_token_id", gen.get("eos_token_id")))
    if bos is not None:
        rows.append(("tokenizer.ggml.bos_token_id", str(int(bos)), "string"))
    if eos is not None:
        rows.append(("tokenizer.ggml.eos_token_id", str(int(eos)), "string"))
    return rows


def _chat_template(model_dir: Path) -> list[tuple[str, str, str]]:
    """A sibling chat template, which `--chat` needs and safetensors splits out.

    Recent exports write it beside the model as ``chat_template.jinja``
    rather than inside ``tokenizer_config.json``; both are read here so
    `--chat` wraps the prompt the way the model was tuned for instead of
    handing it a bare completion.
    """
    path = model_dir / "chat_template.jinja"
    if path.exists():
        try:
            return [("tokenizer.chat_template", path.read_text(), "string")]
        except OSError:
            return []

    config_path = model_dir / "tokenizer_config.json"
    if config_path.exists():
        try:
            template = json.loads(config_path.read_text()).get("chat_template")
        except (json.JSONDecodeError, OSError):
            return []
        if isinstance(template, str):
            return [("tokenizer.chat_template", template, "string")]
    return []


def _default_db_path(src: Path) -> Path:
    """Where to put the database when the caller did not say.

    Named after the model and written beside it, never inside a model
    directory -- a stray .db in there would be picked up by the next glob.
    """
    if src.is_dir():
        return src.parent / f"{src.name}.db"
    if src.name.endswith(".index.json"):
        # The model's identity is its directory, not the index filename.
        return src.parent.parent / f"{src.parent.name}.db"
    return src.with_suffix(".db")


def _detect_mlx_affine(shards: list[Path]) -> bool:
    """True when the model uses MLX-style affine quantization.

    MLX quantizes a weight W as ``(packed_ints, scales, biases)`` where
    ``W ≈ packed * scales + biases``. The three are stored as sibling
    tensors named ``X.weight`` (U32), ``X.scales`` (BF16), ``X.biases``
    (BF16). Detecting even one such triplet is enough.
    """
    names: set[str] = set()
    for shard in shards:
        header, _ = read_header(shard)
        for name, entry in _entries(header).items():
            names.add(name)
    for name in names:
        if (name.endswith(".weight")
                and name[:-7] + ".scales" in names
                and name[:-7] + ".biases" in names):
            return True
    return False


def _infer_mlx_bits(config: dict) -> int:
    """Read the quantization bit width from config.json."""
    qc = config.get("quantization_config") or config.get("quantization") or {}
    return int(qc.get("bits", 4))


def _infer_mlx_group_size(config: dict) -> int:
    """Read the group size from config.json."""
    qc = config.get("quantization_config") or config.get("quantization") or {}
    return int(qc.get("group_size", 64))


def _build_tensor_index(
    shards: list[Path],
) -> dict[str, tuple[Path, int, str, list[int]]]:
    """Map every tensor name to its location without reading data.

    Returns ``{name: (shard_path, data_start, dtype, shape)}``.
    """
    index = {}
    for shard in shards:
        header, data_start = read_header(shard)
        for name, entry in _entries(header).items():
            dtype = entry["dtype"]
            if dtype not in SAFETENSORS_DTYPES:
                raise ValueError(
                    f"Tensor '{name}' in {shard.name} has unknown dtype '{dtype}'"
                )
            index[name] = (shard, data_start, dtype, [int(x) for x in entry["shape"]],
                           entry["data_offsets"])
    return index


def _read_one_tensor(
    shard: Path, data_start: int, offsets: list[int],
) -> bytes:
    """Read a single tensor's bytes from a shard."""
    start, end = offsets
    with open(shard, "rb") as f:
        f.seek(data_start + start)
        blob = f.read(end - start)
    if len(blob) != end - start:
        raise ValueError(
            f"{shard.name} ends early: expected {end - start} bytes, got {len(blob)}"
        )
    return blob


def safetensors_to_sqlite(
    input_path: str, db_path: str | None = None, verbose: bool = True
) -> str:
    """Convert a safetensors model (single file, sharded, or a directory) to SQLite.

    Args:
        input_path: A .safetensors file, a model.safetensors.index.json, or a
            directory holding either.
        db_path: Output database path. Defaults to the model name with .db.
        verbose: Print progress.

    Returns:
        Path to the created database.
    """
    from reminis.converter import SCHEMA

    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Not found: {src}")

    shards, index_meta = resolve_shards(src)
    model_dir = shards[0].parent

    if db_path is None:
        db_path = str(_default_db_path(src))
    db_path = str(db_path)

    if verbose:
        print(f"Reading {src} ({len(shards)} shard{'s' if len(shards) != 1 else ''}) ...")

    from reminis.db import INSERT_OR_REPLACE_TENSOR, open_for_bulk_write

    t0 = time.time()
    Path(db_path).unlink(missing_ok=True)
    conn = open_for_bulk_write(db_path)
    conn.executescript(SCHEMA)
    conn.execute("BEGIN")

    meta_rows = [
        ("reminis.source_format", "safetensors", "string"),
        ("reminis.dtype_system", DTYPE_SYSTEM_SAFETENSORS, "string"),
        ("reminis.shards", str(len(shards)), "string"),
    ]

    # config.json values keep their JSON types so export can rebuild the file
    # byte-for-byte-equivalent rather than turning every number into a string.
    config = _read_config(model_dir)
    for key, value in config.items():
        meta_rows.append((f"config.{key}", json.dumps(value), "json"))

    from reminis import arch as arch_registry

    arch_list = config.get("architectures") or []
    hf_arch = arch_list[0] if arch_list else config.get("model_type", "")
    resolved_name, resolved_arch = arch_registry.from_hf_name(str(hf_arch))
    if resolved_name:
        meta_rows.append(("general.architecture", resolved_name, "string"))
        meta_rows.extend(resolved_arch.translate_config(config))
        meta_rows.extend(_tokenizer_meta_from_config(config, resolved_arch))
        meta_rows.extend(_chat_template(model_dir))
    elif hf_arch:
        meta_rows.append(("general.architecture", str(hf_arch), "string"))
    if model_dir.name:
        meta_rows.append(("general.name", model_dir.name, "string"))

    for key, value in index_meta.items():
        meta_rows.append((f"index.{key}", json.dumps(value), "json"))

    meta_rows.extend(_read_tokenizer(model_dir))

    total_tensors = 0
    total_bytes = 0

    # Collect shard-level metadata.
    for shard in shards:
        header, _ = read_header(shard)
        for key, value in (header.get("__metadata__") or {}).items():
            meta_rows.append((f"st.{key}", str(value), "string"))

    # Detect MLX-style affine quantization and build a tensor index
    # (location only, no data loaded yet).
    is_mlx_affine = _detect_mlx_affine(shards)
    tensor_index = _build_tensor_index(shards)
    mlx_bits = _infer_mlx_bits(config) if is_mlx_affine else 4

    if is_mlx_affine:
        meta_rows.append(("reminis.mlx_affine_bits",
                          str(mlx_bits), "string"))
        meta_rows.append(("reminis.mlx_affine_group_size",
                          str(_infer_mlx_group_size(config)), "string"))

    for name in sorted(tensor_index):
        shard, data_start, dtype, logical_shape, offsets = tensor_index[name]
        blob = _read_one_tensor(shard, data_start, offsets)

        n_elements = 1
        for d in logical_shape:
            n_elements *= d

        # Translate name.
        out_name = name
        if resolved_arch is not None:
            translated = resolved_arch.translate_name(name)
            if translated is not None:
                out_name = translated

        # Apply tensor-level transforms (e.g. A_log → -exp(A_log)).
        if resolved_arch is not None and resolved_arch.needs_transform(out_name):
            arr = to_float32(blob, dtype).reshape(logical_shape)
            arr = resolved_arch.transform_tensor(out_name, arr)
            logical_shape = list(arr.shape)
            n_elements = arr.size
            blob = from_float32(arr, dtype)
            del arr

        conn.execute(
            INSERT_OR_REPLACE_TENSOR,
            (
                out_name,
                json.dumps(logical_shape[::-1]),
                dtype,
                SAFETENSORS_DTYPE_IDS[dtype],
                n_elements,
                len(blob),
                blob,
            ),
        )
        del blob
        total_tensors += 1
        total_bytes += n_elements * SAFETENSORS_DTYPES[dtype]

        if verbose:
            print(
                f"  [{total_tensors}] {out_name:60s} {str(logical_shape):20s} "
                f"{dtype:6s} {human_bytes(n_elements * SAFETENSORS_DTYPES[dtype])}"
            )

    conn.executemany(
        "INSERT OR REPLACE INTO model_meta (key, value, type) VALUES (?, ?, ?)",
        meta_rows,
    )

    conn.execute("COMMIT")
    conn.close()

    elapsed = time.time() - t0
    total_mb = total_bytes / (1024 * 1024)
    if verbose:
        print(f"\nDone. {total_tensors} tensors ({total_mb:.1f} MB) stored in {db_path}")
        if elapsed > 0:
            print(f"Time: {elapsed:.1f}s ({total_mb / elapsed:.1f} MB/s)")

    return db_path


def sqlite_to_safetensors(
    db_path: str, output_path: str | None = None, verbose: bool = True
) -> str:
    """Write a reminis database back out as a single safetensors file.

    A sibling config.json is written when the database carries ``config.*``
    metadata, so the output directory is loadable by transformers.

    GGUF-sourced databases can be exported too, provided every tensor uses a
    dtype safetensors has -- quantized GGML types do not, and are refused.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    if output_path is None:
        output_path = str(db_path.with_suffix(".safetensors"))
    output_path = str(output_path)

    t0 = time.time()
    conn = sqlite3.connect(str(db_path))
    system = dtype_system(conn)

    rows = conn.execute(
        "SELECT name, shape, dtype, n_elements, n_bytes FROM tensors ORDER BY id"
    ).fetchall()
    if not rows:
        conn.close()
        raise ValueError(f"{db_path} contains no tensors")

    if system == DTYPE_SYSTEM_GGUF:
        unsupported = sorted(
            {r[2] for r in rows if r[2] not in GGML_TO_SAFETENSORS_DTYPE}
        )
        if unsupported:
            conn.close()
            raise ValueError(
                "Cannot export to safetensors: the model uses GGML types with no "
                f"safetensors equivalent ({', '.join(unsupported)}). Quantized "
                "GGUF models must be exported as GGUF."
            )

    # Build the header first: safetensors puts every offset up front, so the
    # full layout has to be known before any bytes are written.
    header = {}
    offset = 0
    plan = []
    for name, shape_json, dtype, n_elements, n_bytes in rows:
        out_dtype = (
            GGML_TO_SAFETENSORS_DTYPE[dtype] if system == DTYPE_SYSTEM_GGUF else dtype
        )
        if out_dtype not in SAFETENSORS_DTYPES:
            conn.close()
            raise ValueError(f"Tensor '{name}' has dtype '{dtype}', unknown to safetensors")
        logical_shape = json.loads(shape_json)[::-1]
        header[name] = {
            "dtype": out_dtype,
            "shape": logical_shape,
            "data_offsets": [offset, offset + n_bytes],
        }
        plan.append((name, n_bytes))
        offset += n_bytes

    st_meta = {
        key[len("st.") :]: value
        for key, value in conn.execute(
            "SELECT key, value FROM model_meta WHERE key LIKE 'st.%'"
        )
    }
    st_meta.setdefault("format", "pt")
    st_meta["reminis_version"] = _version()
    header["__metadata__"] = st_meta

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # The spec wants the data block 8-byte aligned; pad the header with spaces,
    # which JSON ignores.
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad

    with open(output_path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        # Read ahead on other threads while this one writes; the header has
        # already fixed the order these have to land in. See
        # `db.read_blobs_ahead` for why the read half parallelises and the
        # write half does not.
        blobs = read_blobs_ahead(str(db_path), [name for name, _ in plan])
        for i, ((name, n_bytes), (_, blob)) in enumerate(zip(plan, blobs)):
            f.write(blob)
            del blob
            if verbose:
                print(f"  [{i+1}/{len(plan)}] {name:60s} {n_bytes / 1024:10.1f} KB")

    config = {}
    for key, value in conn.execute(
        "SELECT key, value FROM model_meta WHERE key LIKE 'config.%'"
    ):
        try:
            config[key[len("config.") :]] = json.loads(value)
        except json.JSONDecodeError:
            config[key[len("config.") :]] = value
    if config:
        (Path(output_path).parent / "config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )

    conn.close()

    total_mb = offset / (1024 * 1024)
    elapsed = time.time() - t0
    if verbose:
        print(f"\nDone. {len(plan)} tensors ({total_mb:.1f} MB) written to {output_path}")
        if config:
            print(f"Wrote config.json alongside it ({len(config)} keys)")
        if elapsed > 0:
            print(f"Time: {elapsed:.1f}s ({total_mb / elapsed:.1f} MB/s)")

    return output_path


def load_tensors(input_path: str) -> dict:
    """Read a safetensors model into ``{name: (dtype, logical_shape, bytes)}``.

    Used by the LoRA path, where adapters are small enough to hold in memory.
    """
    shards, _ = resolve_shards(Path(input_path))
    out = {}
    for shard in shards:
        header, data_start = read_header(shard)
        with open(shard, "rb") as f:
            for name, entry in sorted(
                _entries(header).items(), key=lambda kv: kv[1]["data_offsets"][0]
            ):
                start, end = entry["data_offsets"]
                f.seek(data_start + start)
                out[name] = (entry["dtype"], [int(x) for x in entry["shape"]], f.read(end - start))
    return out


def _version() -> str:
    from reminis import __version__

    return __version__
