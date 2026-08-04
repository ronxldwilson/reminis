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

# Browse it in your browser
reminis view model.db

# Compare two models, and package the difference
reminis diff base.db finetuned.db -o change.delta.db

# Reconstruct the fine-tune from the base plus the pack
reminis apply base.db change.delta.db -o rebuilt.db

# Convert back to GGUF
reminis export model.db -o model_restored.gguf
```

## Verified Results

SHA256-verified lossless round-trip across 9 model variants covering 13 quantization types:

```
Model                               Dtypes                Tensors  GGUF MB    DB MB    RT MB  Conv(s)   Exp(s)   Result
---------------------------------------------------------------------------------------------------------------------------------------
SmolLM-135M.IQ3_M                   F32,IQ3_S,IQ4_NL,Q4_K       272     86.0     86.1     84.3     1.73     0.05     PASS
SmolLM-135M.IQ4_XS                  F32,IQ4_NL,IQ4_XS,Q5_K      272     87.1     87.1     85.4     1.81     0.06     PASS
SmolLM-135M.Q2_K                    F32,IQ4_NL,Q3_K,Q8_0         272     84.1     84.2     82.4     1.64     0.05     PASS
SmolLM-135M.Q3_K_M                  F32,IQ4_NL,Q4_K,Q5_0         272     89.2     89.3     87.5     1.67     0.06     PASS
SmolLM-135M.Q4_K_M                  F32,Q4_K,Q5_0,Q6_K,Q8_0      272    100.6    100.7     98.9     1.71     0.05     PASS
SmolLM-135M.Q5_K_M                  F32,Q5_1,Q5_K,Q6_K,Q8_0      272    106.9    106.9    105.2     1.83     0.07     PASS
SmolLM-135M.Q6_K                    F32,Q6_K,Q8_0                 272    132.0    132.0    130.3     1.84     0.07     PASS
SmolLM-135M.Q8_0                    F32,Q8_0                      272    138.1    138.2    136.4     1.92     0.08     PASS
SmolLM-135M.f16                     F16,F32                       272    258.3    258.4    256.7     2.39     0.14     PASS
---------------------------------------------------------------------------------------------------------------------------------------
9/9 models passed SHA256-verified lossless round-trip
ALL TESTS PASSED - every tensor in every model matches byte-for-byte
```

Every tensor in every model was hashed with SHA256 before and after the round-trip. Zero data loss.

## What's in the Database

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

Every tensor gets its own row with full metadata:

| Column | Description |
|--------|-------------|
| `name` | Tensor path (e.g. `blk.5.attn_q.weight`) |
| `shape` | Dimensions as JSON (e.g. `[576, 576]`) |
| `dtype` | Data type (`F16`, `F32`, `Q4_K`, `Q8_0`, etc.) |
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

## Diffing and Delta Packs

`reminis diff` compares two models tensor by tensor and can emit a **delta pack** — a small database that reconstructs the target from the base:

```bash
reminis diff base.db instruct.db -o change.delta.db
reminis apply base.db change.delta.db -o rebuilt.db
```

Packs record the weight hashes of both sides. `apply` refuses a base that does not match, rather than silently producing a corrupt model, and verifies the result against the recorded target hash.

Encoding is **XOR**, not arithmetic subtraction. An arithmetic float delta is not exactly reversible — `b - a` generally is not representable in the tensor's own dtype, so `a + delta` lands a rounding step away from `b`. XOR is exact for every dtype, and also works on quantized tensors, whose bytes cannot be subtracted meaningfully at all. Per tensor, whichever of the compressed XOR delta or a compressed full replacement is smaller wins.

### How small are packs, really?

It depends entirely on how much of the model the fine-tune touched.

| Scenario | Tensors changed | Pack size |
|---|---|---|
| Targeted change (5 of 272 tensors) | 5 | **1.4%** of full model |
| Full fine-tune (SmolLM-135M to its Instruct variant) | 272 of 272 | **58.8%** of full model |

The second row is the honest one for full fine-tuning. Every tensor changed, with ~97% of individual values differing in each, so lossless compression has little to exploit — the differing float16 mantissa bits are close to incompressible.

The delta is also only partly low-rank. Ranks needed to capture 90% of the delta's energy:

| Tensor | Shape | Rank for 90% | Full rank |
|---|---|---|---|
| `blk.0.attn_q.weight` | [576, 576] | 31 | 576 |
| `blk.18.ffn_gate.weight` | [1536, 576] | 311 | 576 |
| `blk.29.ffn_down.weight` | [576, 1536] | 291 | 576 |

Attention deltas compress well under a low-rank factorization; FFN deltas do not. Lossy encodings that would change this — int8 deltas, adaptive-rank SVD, mantissa truncation — are tracked in [issue #2](https://github.com/ronxldwilson/reminis/issues/2) and are not implemented yet.

## Python API

```python
from reminis import gguf_to_sqlite, sqlite_to_gguf

# Convert
db_path = gguf_to_sqlite("model.gguf")

# Query with standard sqlite3
import sqlite3
conn = sqlite3.connect(db_path)
for name, n_elements in conn.execute(
    "SELECT name, n_elements FROM tensors ORDER BY n_elements DESC LIMIT 5"
):
    print(f"{name}: {n_elements:,} params")

# Export back
sqlite_to_gguf(db_path, "model_restored.gguf")
```

## Supported Formats

All GGUF tensor types are supported and verified, including:

| Type | Description | Verified |
|------|-------------|----------|
| F32, F16 | Full precision | Yes |
| Q4_K, Q5_K, Q6_K | K-quants (4/5/6 bit) | Yes |
| Q8_0 | 8-bit quantized | Yes |
| Q3_K, Q5_0, Q5_1 | Other quants | Yes |
| IQ3_S, IQ4_NL, IQ4_XS | Importance-weighted quants | Yes |

## Roadmap

- [x] Publish to PyPI
- [x] GGUF to SQLite converter (lossless, verified across 13 quant types)
- [x] SQLite to GGUF back-converter (lossless, byte-perfect)
- [x] SHA256 verification test suite
- [x] Interactive HTML viewer (`reminis view`)
- [x] Weight diffing between model versions (`reminis diff`)
- [x] Delta packs with verified apply (`reminis apply`)
- [ ] Lossy delta encodings so full fine-tunes compress ([#2](https://github.com/ronxldwilson/reminis/issues/2))
- [ ] Fine-tune tracking with edit logs
- [ ] Surgical rollback of bad training steps
- [ ] Model merging via SQL operations
- [ ] Inference from database-stored weights
- [ ] Unsloth integration

## License

MIT
