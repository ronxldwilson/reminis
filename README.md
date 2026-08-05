# reminis

**Your model's weights are just data. Store them in a database.**

`reminis` converts any GGUF or safetensors model into a SQLite database where every tensor becomes a queryable, versionable, diffable row — one you can also merge with a join and run text through. Convert back when you're done. Lossless. Fast.

```bash
pip install reminis
```

## Why

The ML world treats model weights as opaque files. You save the whole thing, load the whole thing, and if something goes wrong, you retrain from scratch.

Once weights are in a database, you get — for free — much of what 40 years of database engineering has built: queries, diffs, snapshots, audit logs, replication.

Not all of it, and reminis tries to be specific about which. Rollback, in particular, does not mean what it means in a database — see [what rollback actually gives you](#what-rollback-actually-gives-you--a-negative-result).

The database is also still a *model*, not an archive of one: `reminis run` generates text straight from the rows, and its logits match `transformers` to 4e-05.

## Quick Start

```bash
# Convert a GGUF model to SQLite
reminis convert model.gguf

# ...or a safetensors model. Point at the file, the index, or the directory;
# sharded checkpoints and config.json are handled for you.
reminis convert ./Llama-3.2-1B/

# Turn a peft LoRA adapter into a delta pack against its base
reminis lora ./my-adapter/ base.db -o capability.pack.db

# Keep a base and everything fine-tuned from it in ONE database
reminis registry add      models.db ./Llama-3.2-1B/ --name llama-1b
reminis registry add-lora models.db ./my-adapter/   --name llama-1b-sql --parent llama-1b
reminis registry ls models.db

# Inspect what's inside
reminis info model.db

# Browse it in your browser
reminis view model.db

# Compare two models, and package the difference
reminis diff base.db finetuned.db -o change.delta.db

# Reconstruct the fine-tune from the base plus the pack
reminis apply base.db change.delta.db -o rebuilt.db

# Merge models: the tensors are aligned by a SQL join
reminis merge base.db instruct.db -o soup.db
reminis merge sql.db chat.db -o both.db --method ties --base base.db

# Generate text straight from the weights in the database
reminis run soup.db "The capital of France is"

# See what a tracked training run did, and rewind to a snapshot
reminis log run.log.db
reminis rollback run.log.db 500 -o restored.db

# Convert back to GGUF
reminis export model.db -o model_restored.gguf
```

## Verified Results

### Architectures

Every tensor is SHA256-hashed before and after the round-trip. These architectures are confirmed lossless:

| Architecture | Model | Tensors | Size | Convert | Export | Result |
|---|---|---|---|---|---|---|
| `llama` | Mistral-7B-Instruct-v0.3 Q4_K_M | 291 | 4170 MB | 18.5s | 4.4s | lossless |
| `llama` | Llama-3.2-1B-Instruct F16 | 147 | 2365 MB | 19.2s | 1.7s | lossless |
| `qwen2` | Qwen2.5-0.5B-Instruct FP16 | 291 | 1208 MB | 14.4s | 0.9s | lossless |
| `granitemoe` | Granite-3.1-1B-A400M Q4_K_M | 242 | 784 MB | 6.5s | 0.5s | lossless |
| `clip` (vision) | SmolVLM-256M projector F16 | 198 | 181 MB | 1.0s | 0.1s | lossless |
| `mamba` | Mamba-130M Q4_K_M | 242 | 86 MB | 1.7s | 0.04s | lossless |
| `rwkv7` | RWKV7-G1h-1.5B Q2_K | 678 | 644 MB | 3.5s | 0.4s | lossless |
| `granitehybrid` | Granite-4.0-H-Micro Q2_K | 506 | 1169 MB | 7.3s | 0.6s | lossless |
| `llama` | SmolLM-135M, 9 quantizations | 272 | 84–258 MB | ~2s | ~0.1s | lossless |

The MoE model is the interesting one: 72 of its tensors are **3-dimensional** expert stacks (e.g. `[512, 1024, 32]` in Q6_K), a shape the quantized export path had never seen. It generalizes correctly.

Mamba and RWKV7 are there because they are not transformers at all — no attention, no feed-forward network — and the RWKV7 file mixes BF16 tensors into a GGUF alongside Q2_K and Q6_K. Granite 4.0-H is a hybrid: 36 state-space blocks with attention at blocks 5, 15, 25 and 35. All round-trip byte-for-byte.

Throughput is roughly linear with size — about 120 MB/s converting, over 1 GB/s exporting.

### The architecture diagram

`reminis view` builds each block out of the tensors that are actually in it, instead of assuming every model is a transformer:

| Family | What a block shows |
|---|---|
| Dense transformer | Self-Attention, then Feed-Forward |
| Mixture of Experts | Self-Attention, then the expert stack and its router |
| Mamba / state space | State Space (SSM) — no attention, because there is none |
| RWKV | Time Mixing, then Channel Mixing |
| Hybrid (attention in some blocks, a scan in others) | whichever of those that block holds |
| Vision tower | patch embedding, position embedding, the blocks, and the projector |
| Anything unrecognised | the block's tensors and their size, with no invented labels |

Long stacks are collapsed, but by runs of identical blocks rather than a fixed first-two/last-two window. That distinction is load-bearing for a hybrid: Granite 4.0-H Micro puts its only four attention blocks at 5, 15, 25 and 35, all of which a fixed window drops — and the collapsed summary would then have claimed "same structure" about the blocks that were the exception.

For a MoE model the diagram also answers the question the file size cannot. Granite-3.1-1B-A400M reads **428.7M of 1.33B parameters run for any one token (32%)** — the router picks 8 of 32 experts, so the remaining 906M sit idle on every forward pass. That 428.7M is a good check on the arithmetic: it is the "A400M" in the model's own name.

Both naming conventions are handled, so a MoE model imported from safetensors (experts stored one per tensor, `num_local_experts` in the config) reports the same numbers as one imported from GGUF (experts stacked into 3-D tensors, `expert_count` in the metadata).

`tests/test_viewer.py` renders the diagram under node with a stub DOM and asserts on what it produces, since the diagram is built by JavaScript at page load and checking the HTML for a string would prove nothing. It skips when node is not on PATH.

### Safetensors

Same SHA256 verification, on real downloaded checkpoints:

| Model | Dtype | Tensors | Size | Convert | Export | Result |
|---|---|---|---|---|---|---|
| SmolLM-135M | F32 | 272 | 513 MB | 1.9s | 0.2s | lossless |
| SmolLM2-135M-Instruct | **BF16** | 272 | 257 MB | 1.0s | 0.1s | lossless |

BF16 is the case that matters: it is what most fine-tuning emits, and it is the reason reminis parses the format directly rather than through `safetensors.numpy`, which cannot load BF16 at all. The conversion is verified bit-identical to PyTorch in both directions, across normal values, subnormals, infinities, and NaN.

### LoRA adapters against peft

A rank-16 adapter on SmolLM2-135M (BF16 base, all 7 projection types targeted, 210 modules), converted to a pack, applied, and compared tensor-by-tensor against peft's own `merge_and_unload()`:

| Check | Result |
|---|---|
| Tensors byte-identical to peft's merge | **272 / 272** |
| Worst relative difference | 0.000e+00 |
| Worst gap in BF16 representable steps | 0 |
| Pack size | 17.3 MB (6.7% of the 256.6 MB base) |

Bit-exact agreement with peft, on a BF16 base where rounding could have shown up and did not.

### Diff and apply at scale

Perturbing all 64 attention projections, then reconstructing from the pack:

| Model | Copy DB | Diff | Pack | Apply + verify | Result |
|---|---|---|---|---|---|
| Llama-3.2-1B, 2.4 GB, F16 | 0.57s | 9.4s | 278 MB (11.8%) | 7.9s | exact |
| Mistral-7B, 4.2 GB, Q4_K_M | 1.78s | 14.5s | 15 MB (0.4%) | 12.2s | exact |

The 7B row is quantized, so per-value deltas are not computable — but changes are still detected and encoded byte-exactly through the XOR path, which is why quantized models work at all.

Roughly half of each timing is SHA256 hashing the full model, which is what guarantees a pack cannot be applied to the wrong base. SHA256 is hardware-accelerated here (957 MB/s) and measurably faster than blake2b, so that is already the cheapest safe option.

### Inference against a reference implementation

`reminis run` computes its forward pass in numpy from tensors selected out of SQLite. To show that this is the same model rather than a plausible imitation, its output is checked against `transformers` running the original checkpoint in float32:

| Check | Result |
|---|---|
| Tokenizer ids vs `transformers`, 16 strings × 3 tokenizer families | **48 / 48 identical** |
| Top-10 next tokens, SmolLM-135M | identical, in order |
| Largest difference in any of the 49,152 logits | **4.0e-05** |
| Token-by-token decoding vs one batched pass | 3.4e-05 |

The logit gap is the F16 the GGUF stores against the F32 torch loads, which is the size it should be. The last row uses no external reference at all: if the KV cache, the rotary positions, or the causal mask were wrong, incremental decoding would diverge from a single batched pass, and it does not.

### Inference speed

Greedy decoding, single-threaded numpy, on an M-series laptop:

| Model | Database | Prompt | Generation |
|---|---|---|---|
| SmolLM-135M f16 | 258 MB | 0.38s | 80.1 tok/s |
| Qwen2.5-0.5B f16 | 1,208 MB | 1.99s | 26.4 tok/s |
| Llama-3.2-1B f16 | 2,366 MB | 4.93s | 7.6 tok/s |
| SmolLM-135M f16, `--stream` | 258 MB | 0.37s | 2.8 tok/s |

This is not fast, and it is not trying to be — llama.cpp will beat it by an order of magnitude and always will. The last row is the interesting one: with `--stream`, nothing is cached, so those 8 tokens re-read **2,796 MB across 2,457 queries** from a 258 MB file. Peak memory is one layer instead of one model.

### Quantization coverage

SHA256-verified lossless round-trip across 9 SmolLM-135M variants covering 13 quantization types:

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
| `name` | Tensor path (e.g. `blk.5.attn_q.weight`, or `model.layers.5.self_attn.q_proj.weight` from safetensors) |
| `shape` | Dimensions as JSON (e.g. `[576, 576]`) |
| `dtype` | Data type (`F16`, `F32`, `BF16`, `Q4_K`, `Q8_0`, etc.) |
| `dtype_id` | Numeric type id — see the note below |
| `n_elements` | Number of parameters |
| `n_bytes` | Storage size in bytes |
| `data` | Raw weight data as BLOB |

`shape` is stored **reversed relative to the data layout**, which is GGUF's convention; reminis keeps it for every format so one set of code reads both. A safetensors tensor of logical shape `[out, in]` is stored as `[in, out]` and un-reversed on export.

`dtype_id` means different things in different databases — a GGML enum value for GGUF-sourced models, a reminis-local id for safetensors ones — so `model_meta` records which system applies under `reminis.dtype_system`. Read the `dtype` name unless you specifically need the id.

All model metadata (architecture, context length, vocab size, etc.) is stored in a `model_meta` table. For safetensors models this is populated from the sibling `config.json` under `config.*` keys, since the format itself carries almost no metadata of its own.

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

Attention deltas compress well under a low-rank factorization; FFN deltas do not. That asymmetry is exactly what `--lossy` exploits.

### Low-rank packs for LoRA fine-tunes

A LoRA update is `W + BA` with `BA` rank-`r` by construction, so the delta is genuinely low-rank and needs only `r*(m+n)` numbers instead of `m*n`. Opt in with `--lossy`:

```bash
reminis diff base.db lora-merged.db -o change.delta.db --lossy 0.01
```

The tolerance is the maximum relative error allowed per tensor (default `0.01` = 1%). Measured on a rank-16 LoRA merge:

| Model | Lossless pack | Low-rank pack | Shrink | Worst error |
|---|---|---|---|---|
| SmolLM-135M, 258 MB | 38.3 MB (14.9%) | **3.5 MB (1.4%)** | 11.0x | 1.2e-04 |
| Llama-3.2-1B, 2.4 GB | 242.7 MB (10.3%) | **6.4 MB (0.3%)** | 37.9x | 1.1e-04 |

Achieved error lands ~100x inside the 1% budget, because rank is chosen by error target rather than fixed.

**It decides per tensor, and never makes a pack worse.** Low-rank is only used where it both beats the lossless size and stays inside tolerance; otherwise the lossless encoding is kept. On the full fine-tune above, only 1 of 272 tensors qualified and the pack stayed lossless — so `--lossy` degrades gracefully to "no change" rather than silently hurting quality.

Cost is a slower diff (SVD): roughly 4x on the 1B model. Quantized and non-2D tensors are never low-rank encoded, since their bytes cannot be decomposed.

**Lossy packs are still exactly verifiable.** The reconstruction is deterministic, so the pack records the hash of what `apply` will actually produce, and `apply` checks against it byte-for-byte. What it cannot promise is that this equals the *original* target — that divergence is carried as a recorded error bound and printed on apply:

```
Result verified against reconstruction hash
NOTE: this is a lossy pack (120 tensors low-rank encoded). The result is not
byte-identical to the original target; worst per-tensor relative error is 1.16e-04.
```

## LoRA adapters ship as exact delta packs

A LoRA adapter already *is* a low-rank delta — peft saves `lora_A` and `lora_B`, and the update it applies is `(alpha / r) * B @ A`. That is the same structure as a reminis low-rank pack, except the factors are the real ones rather than an SVD approximation of a finished merge.

So `reminis lora` converts an adapter with no SVD, no merge, and no approximation error:

```bash
reminis lora ./my-adapter/ base.db -o capability.pack.db
reminis apply base.db capability.pack.db -o merged.db
```

Verified against peft itself: the applied result is compared tensor-by-tensor against `merge_and_unload()`, on a toy float32 model and on a real BF16 SmolLM2-135M with 210 targeted modules. **Every tensor came out byte-identical to peft's own merge** in both cases — not merely close. The tests still assert a tolerance rather than byte-equality, since that is the property that actually matters and a mis-applied `alpha / r` would blow past it by orders of magnitude.

`modules_to_save` tensors — ones peft trained outright rather than through a factor pair — are carried in the pack in full. Embedding LoRA (`lora_embedding_A/B`) is not handled yet, and reminis refuses such an adapter rather than writing a pack that quietly omits part of it.

## Many models in one database

A base model and everything fine-tuned from it can live in a single file. Base models store their weights outright; anything derived stores **only what differs**, using the same verified delta encodings as `reminis diff`.

```bash
# A base, then two peft LoRA fine-tunes of it
reminis registry add      models.db ./SmolLM2-135M/  --name smollm2-135m
reminis registry add-lora models.db ./sql-adapter/   --name smollm2-sql  --parent smollm2-135m
reminis registry add-lora models.db ./chat-adapter/  --name smollm2-chat --parent smollm2-135m

reminis registry ls models.db
```

```
  NAME                    KIND     PARENT           TENSORS   FULL SIZE      STORED
  ------------------------------------------------------------------------------------
  smollm2-135m            base     -                    272    256.6 MB    256.6 MB  100.0%
    smollm2-sql           lora     smollm2-135m         272    256.6 MB      3.3 MB    1.3%
    smollm2-chat          lora     smollm2-135m         272    256.6 MB      3.3 MB    1.3%
  ------------------------------------------------------------------------------------

  3 models (1 base, 2 derived)
  Stored separately, these would be: 769.7 MB
  This registry file is:             263.6 MB
  Saved: 65.8%  (506.1 MB)
```

Each fine-tune costs **1.3%** of a full copy. Get any of them back as a real model:

```bash
reminis registry export models.db --name smollm2-sql -o sql.db
reminis registry export models.db --name smollm2-sql -o sql.safetensors
reminis registry export models.db --name smollm2-sql -o sql.gguf
```

Verified: a model exported from a registry is **byte-identical to peft's own `merge_and_unload()`** — all 272 tensors, same SHA256.

Full fine-tunes work too, they just cost more, because that is genuinely how much they differ:

| Model | Stored |
|---|---|
| base (SmolLM-135M) | 256.6 MB — 100% |
| LoRA-shaped fine-tune | 38.3 MB — 14.9% |
| LoRA-shaped fine-tune, `--lossy` | ~3 MB — ~1.3% |
| SmolLM-135M-Instruct (full fine-tune) | 149.5 MB — 58.3% |

Notes on the design:

- Models form a **tree**. A fine-tune of a fine-tune is stored against its immediate parent, and resolving a tensor walks the chain. Tested two levels deep.
- Every model records the SHA256 of its own weights when added, and that hash is **checked at add time** — if storing a model would not reproduce it, nothing is added. It is checked again on export.
- Removing a model that others derive from is refused, rather than leaving them unresolvable.
- Single-model `.db` files are unchanged. A registry is a separate format; `convert` and `export` still produce ordinary one-model files, and a registry ingests or emits them. Pointing `reminis info` at a registry says so instead of silently summing every model together.

```python
from reminis import Registry

with Registry("models.db") as reg:
    reg.add_base("./SmolLM2-135M/", "smollm2-135m")
    reg.add_lora("./sql-adapter/", "smollm2-sql", parent="smollm2-135m")

    for m in reg.list_models():
        print(m["name"], m["stored_bytes"])

    reg.materialize("smollm2-sql", "sql.db")     # back to a normal database
```

## Merging models is a join

Model merging is normally a bespoke script: load two checkpoints into memory, match parameter names by hand, average, save. Once the weights are rows, the matching half of that is a join. `reminis merge` attaches every input onto one SQLite connection and asks which tensor names line up, with which shapes, in which dtypes — and that query *is* the merge plan.

```bash
# Weighted average ("model soup")
reminis merge base.db instruct.db -o soup.db --weights 0.7,0.3

# Spherical interpolation between two checkpoints
reminis merge base.db instruct.db -o slerp.db --method slerp -t 0.35

# Combine two fine-tunes as task vectors against their shared base
reminis merge sql.db chat.db -o both.db --method task-arithmetic --base base.db

# TIES: trim each task vector, elect a sign per parameter, average the agreers
reminis merge sql.db chat.db -o both.db --method ties --base base.db --density 0.2

# One fine-tune at a negative scale subtracts what it learned
reminis merge sql.db -o unlearned.db --method task-arithmetic --base base.db --scale -1
```

| Method | What it does | Needs |
|---|---|---|
| `linear` | Weighted average, normalised to sum 1 | 2+ models |
| `slerp` | Interpolation along the arc, so magnitude travels with direction | exactly 2 |
| `task-arithmetic` | `base + Σ wᵢ·(modelᵢ − base)` | `--base` |
| `ties` | Task vectors trimmed to `--density`, sign elected by total magnitude, then averaged over the models that agree | `--base` |

Because the alignment is declarative, a bad merge fails as a row in a result set rather than as an exception thrown halfway through a written file:

- **Quantized tensors are refused, not approximated.** Averaging two Q4_K blocks byte by byte produces noise that still parses as a model, which is the worst available failure. If most of a model's bytes are quantized but identical in every input, the merge says so rather than reporting a merge that changed 0.4% of the file.
- **Shape mismatches are refused**, with the count and an example — those models are not the same architecture.
- **Dtypes may differ.** An F32 model and a BF16 one merge fine: everything is combined in float32 and written back in the first model's dtype.
- **Tensors only the first model has** are carried through; ones only a later model has are dropped, and both counts are reported.
- **Nothing is written on failure**, and the output may not be one of the inputs.

The result records where it came from — `reminis.merge.method`, `.sources`, `.weights`, `.base` — so a merged file is never anonymous.

Verified on two real 135M checkpoints that share an architecture (SmolLM-135M and its instruct fine-tune, 272 tensors, 134.5M parameters, 1.9s). The sharpest check is an identity: `task-arithmetic` at scale 1 against the base must reconstruct the fine-tune, and it reproduces **every one of the 134,515,008 weights**. Four tensors come back with a different bit pattern for the same number, because adding a zero task vector to a `+0.0` base yields `+0.0` where the original stored `-0.0` — which is what floating-point addition does, and worth stating rather than papering over.

## Running the model out of the database

The obvious question about storing weights in SQLite is whether the result is still a model or just an archive of one. `reminis run` answers it by generating text — a pure-numpy forward pass over tensors selected out of the database, with no torch, no llama.cpp, and no config files. Everything it needs is already in the file: the weights are rows in `tensors`, and the hyperparameters and the tokenizer's vocabulary and merges are rows in `model_meta`.

```bash
reminis run model.db "The capital of France is"

# Greedy, for a reproducible answer
reminis run model.db "The capital of France is" --temp 0 -n 40

# Wrap the prompt in the model's chat template
reminis run model.db "Name three colours." --chat

# Never cache a weight: every matmul re-reads its operand from SQLite
reminis run model.db "The capital of France is" --stream
```

`--stream` is the mode that makes the claim literal. Nothing is held in memory between uses, so peak memory is one layer rather than one model, and a 258 MB database serves 2,796 MB of reads across 2,457 queries to generate eight tokens. It is ~30× slower, and it is the whole thesis in one flag: the model is data, paged in on demand.

This also closes the loop on everything else here. A merged, rolled-back, or delta-applied database can be checked by asking it to speak:

```bash
reminis merge base.db instruct.db -o soup.db
reminis run soup.db "Name three colours." --chat --temp 0
```

Implemented, and enforced rather than assumed:

- **llama-family and qwen2 architectures** — RMSNorm, SwiGLU, grouped-query attention, rotary embeddings. That covers llama, llama 3 (including its per-dimension rotary scaling, which is stored as a tensor rather than as metadata), mistral, qwen2 (which uses the other rotary layout and has QKV biases), and smollm.
- **Float weights** — F32, F16, BF16.
- **Byte-level BPE**, rebuilt from the vocabulary and merge list in the database, matching `transformers` exactly across three tokenizer families.

Anything else raises. A state-space model is refused by name; a quantized tensor is refused before it can be decoded as though its blocks were floats. A forward pass that guesses produces fluent nonsense, which is worse than an error.

## Tracking a training run

`reminis` can record what training did to a model as it happens, so a bad step can be found later with a query rather than a guess.

```python
from reminis import TrainingLog
from reminis.integrations import TrackedOptimizer

log = TrainingLog("run.log.db", run_name="my-finetune", snapshot_dir="snapshots/")
optimizer = TrackedOptimizer(
    torch.optim.AdamW(model.parameters(), lr=1e-4),
    log,
    model.named_parameters(),
    every_n_steps=5,          # the main cost dial
)

for step, batch in enumerate(loader):
    if step % 500 == 0:
        log.snapshot(step, model.state_dict())
    loss = model(**batch).loss
    loss.backward()
    optimizer.current_loss = float(loss.detach())
    optimizer.step()
    optimizer.zero_grad()
```

With HuggingFace `Trainer`, pass the same wrapped optimizer via `optimizers=` and add `reminis.integrations.make_callback(log, snapshot_every=500)` so loss, epoch, and learning rate reach the log too.

Then read it:

```bash
$ reminis log run.log.db

  Steps logged: 20
  Parameter updates: 420
  Snapshots: 3

  Loss: 4.1679 (step 0) -> 2.8655 (step 19), best 2.8186 at step 18

  Most-updated parameters (by cumulative gradient norm):
    model.embed_tokens.weight                    24.38  over 20 steps
    lm_head.weight                               23.55  over 20 steps

  Snapshots (1.1 MB total):
    step     0  full                  396.0 KB
    step     8  delta  vs step 0      364.0 KB
```

`reminis log run.log.db --step N` shows per-parameter detail for one step. Snapshots are stored as delta packs against the previous snapshot, and `reminis rollback run.log.db <step> -o restored.db` restores one, verified against the hash recorded during training.

### What tracking costs

Measured on SmolLM2-135M (134.5M parameters, fp32, CPU):

| Setting | Per step | Overhead |
|---|---|---|
| untracked | 405 ms | — |
| `every_n_steps=1` | 536 ms | +32% |
| `every_n_steps=5` | 407 ms | +0.5% |

The work is a handful of reductions per parameter, done in the framework rather than by copying to numpy, so on a GPU it costs far less than these CPU figures suggest. `track_params` (log only the trainable subset, e.g. LoRA) and `track_weights=False` cut it further.

### What rollback actually gives you — a negative result

The appealing idea is surgical: find the bad step, subtract its update from the final weights, keep everything learned since. **That does not work, and reminis says so rather than shipping it.**

`tests/experiment_rollback.py` measures it directly. The same model is trained twice on identical data with identical seeds, differing only in whether one step sees a corrupted batch — so run B is the ground truth we would want to recover. Then three ways of "undoing" run A's bad step are compared against it:

| Approach | Distance from ground truth |
|---|---|
| Do nothing (keep the bad step) | 2.19e-02 |
| Subtract the bad step's weight delta | 2.28e-02 — **worse** |
| Rewind to the snapshot before it | exact at that step, but discards the 20 steps since |

Subtracting the delta made it *worse*, not better. Across 3 seeds × 3 step positions the pattern is consistent and matches the theory: it only helps when few steps follow the bad one, and even then by a few percent.

The reason is that every gradient after the bad step was computed *from the weights that step produced*. Those later updates are only valid in the context of the step you want gone. Adam's moment estimates diverge too, and subtracting a weight delta never touches them.

So rollback in reminis is an honest rewind: it restores a snapshot exactly, hash-verified, and tells you plainly that the steps after it are gone rather than selectively removed. Genuinely dropping a mid-run step means rewinding and training forward again — the log tells you *where* to rewind to, which is the part that was previously guesswork.

## Python API

```python
from reminis import gguf_to_sqlite, safetensors_to_sqlite, sqlite_to_gguf

# Convert, from either format
db_path = gguf_to_sqlite("model.gguf")
db_path = safetensors_to_sqlite("./Llama-3.2-1B/")

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

```python
from reminis.merge import merge_models
from reminis.infer import generate

summary = merge_models(["base.db", "instruct.db"], "soup.db", method="linear")
print(summary["tensors_merged"], summary["mean_drift"])

result = generate("soup.db", "The capital of France is", max_tokens=32, temperature=0.0)
print(result["completion"])
```

## Supported Formats

### GGUF

All GGUF tensor types are supported and verified, including:

| Type | Description | Verified |
|------|-------------|----------|
| F32, F16 | Full precision | Yes |
| Q4_K, Q5_K, Q6_K | K-quants (4/5/6 bit) | Yes |
| Q8_0 | 8-bit quantized | Yes |
| Q3_K, Q5_0, Q5_1 | Other quants | Yes |
| IQ3_S, IQ4_NL, IQ4_XS | Importance-weighted quants | Yes |

### Safetensors

Read and written with numpy alone — the format is an 8-byte header length, a JSON header, and raw bytes, so no extra dependency is needed. Single-file, sharded (`model.safetensors.index.json`), and directory inputs all work, and a sibling `config.json` is ingested into `model_meta` and rebuilt on export.

**BF16 is fully supported**, which is the point: it is what most fine-tuning produces, and `safetensors.numpy` cannot load it at all. reminis converts BF16 with round-to-nearest-even, verified bit-identical to PyTorch in both directions.

Note that GGUF and safetensors use different tensor names (`blk.0.attn_q.weight` vs `model.layers.0.self_attn.q_proj.weight`) and different `dtype_id` spaces, so:

- Exporting a safetensors-sourced database as GGUF is refused, rather than writing a file whose dtype ids mean something else.
- Exporting a GGUF-sourced database as safetensors works when every dtype has an equivalent (F32/F16/BF16 and the integer types). Quantized GGML types have none, and are refused.
- Diffing a GGUF base against a safetensors fine-tune is not supported; the fingerprint check catches it and fails clearly.

## Roadmap

- [x] Publish to PyPI
- [x] GGUF to SQLite converter (lossless, verified across 13 quant types)
- [x] SQLite to GGUF back-converter (lossless, byte-perfect)
- [x] SHA256 verification test suite
- [x] Interactive HTML viewer (`reminis view`)
- [x] Weight diffing between model versions (`reminis diff`)
- [x] Delta packs with verified apply (`reminis apply`)
- [x] Validated to 7B across llama / qwen2 / granitemoe / clip / mamba / rwkv7
- [x] Low-rank delta encoding for LoRA fine-tunes (`--lossy`)
- [x] Safetensors input and output, sharded and BF16 (`reminis convert ./model/`)
- [x] peft LoRA adapters as exact delta packs (`reminis lora`)
- [x] Fine-tune tracking with edit logs (`reminis log`)
- [x] Hash-verified rewind to a snapshot (`reminis rollback`)
- [x] Measured why *surgical* rollback of a mid-run step does not work
- [x] Many models in one database, derived ones stored as deltas (`reminis registry`)
- [x] MoE expert routing in the viewer: expert count, router, active-vs-total parameters
- [x] Attention-free architectures in the viewer (Mamba / state space, RWKV, hybrids)
- [x] Model merging via SQL operations (`reminis merge`: linear, slerp, task arithmetic, TIES)
- [x] Inference from database-stored weights (`reminis run`, verified against transformers)
- [ ] Running quantized models directly, without an F16 conversion first
- [ ] Running attention-free architectures (Mamba / state space, RWKV)
- [ ] Unsloth integration

## License

MIT
