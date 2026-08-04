# reminis

**Your model's weights are just data. Store them in a database.**

`reminis` converts any GGUF model into a SQLite database where every tensor becomes a queryable, versionable, diffable row. Convert back to GGUF when you're done. Lossless. Fast.

```bash
pip install reminis
```

## Why

The ML world treats model weights as opaque files. You save the whole thing, load the whole thing, and if something goes wrong, you retrain from scratch.

Once weights are in a database, you get — for free — everything that 40 years of database engineering has built: queries, rollback, diffs, branching, merging, audit logs, access control, replication.

## Quick Start

```bash
# Convert a GGUF model to SQLite
reminis convert model.gguf

# Inspect what's inside
reminis info model.db

# Convert back to GGUF
reminis export model.db -o model_restored.gguf
```

## Verified Results

Tested on SmolLM-135M (134.5M parameters, F16 unquantized):

| Operation | Time | Speed | Result |
|-----------|------|-------|--------|
| GGUF to SQLite | 2.7s | 95 MB/s | 272 tensors, 256.6 MB stored |
| SQLite to GGUF | 0.2s | 1142 MB/s | Byte-perfect reconstruction |

```
$ reminis info model.db

Database: model.db (258.4 MB)
  general.name: SmolLM 135M
  general.architecture: llama
  Metadata fields: 40
  Tensors: 272
  Parameters: 134,515,008
  Weight data: 256.6 MB

  Dtype breakdown:
    F16          211 tensors     256.5 MB
    F32           61 tensors       0.1 MB
```

**Round-trip is lossless** — all 272 tensors match byte-for-byte after GGUF to SQLite to GGUF conversion.

## What's in the Database

Every tensor gets its own row with full metadata:

| Column | Description |
|--------|-------------|
| `name` | Tensor path (e.g. `blk.5.attn_q.weight`) |
| `shape` | Dimensions as JSON (e.g. `[576, 576]`) |
| `dtype` | Data type (`F16`, `F32`, `Q4_K`, etc.) |
| `n_elements` | Number of parameters |
| `n_bytes` | Storage size in bytes |
| `data` | Raw weight data as BLOB |

All model metadata (architecture, context length, vocab size, etc.) is stored in a `model_meta` table.

## Query Your Model

Once in SQLite, you can query weights like any database:

```sql
-- Largest tensors by parameter count
SELECT name, n_elements, n_bytes / 1024 / 1024 as mb
FROM tensors ORDER BY n_elements DESC LIMIT 5;

-- All attention weights in layer 5
SELECT name, shape, dtype FROM tensors
WHERE name LIKE 'blk.5.attn%';

-- Total size by dtype
SELECT dtype, COUNT(*) as count, SUM(n_bytes) / 1024 / 1024 as total_mb
FROM tensors GROUP BY dtype;

-- Model architecture
SELECT key, value FROM model_meta
WHERE key LIKE '%context_length%' OR key LIKE '%block_count%';
```

## Supported Formats

All GGUF tensor types are supported, including:

| Type | Description |
|------|-------------|
| F32, F16, BF16 | Full precision (lossless round-trip verified) |
| Q4_0, Q4_1, Q4_K | 4-bit quantized |
| Q5_0, Q5_1, Q5_K | 5-bit quantized |
| Q8_0, Q8_1, Q8_K | 8-bit quantized |
| Q2_K, Q3_K, Q6_K | Other K-quants |
| IQ1_S, IQ2_S, IQ3_S, IQ4_NL | i-quants |

## Roadmap

- [x] Publish to PyPI
- [x] GGUF to SQLite converter (lossless, 95 MB/s)
- [x] SQLite to GGUF back-converter (lossless, 1142 MB/s)
- [ ] Fine-tune tracking with edit logs
- [ ] Surgical rollback of bad training steps
- [ ] Weight diffing between model versions
- [ ] Delta-based model distribution (weight migrations)
- [ ] Model merging via SQL operations
- [ ] Inference from database-stored weights
- [ ] Unsloth integration

## License

MIT
