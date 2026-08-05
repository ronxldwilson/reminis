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

## What this is, and what it is not

**reminis is a storage and version-control layer for model weights.** It is where models live between training runs: queried, diffed, merged, versioned, packaged as deltas, and exported when you need to actually use one.

**It is not a runtime, and `reminis run` is not a way to serve a model.** It exists so a database can be checked by asking it to speak — after a merge, a rollback, or a delta apply, a hash tells you the bytes changed but cannot tell you the result is still a working model. That is a testing tool, and a genuinely useful one. It is not an inference engine, and three things stop it from becoming one:

- It runs quantized weights by *unpacking* them, so a quantized model becomes runnable but not small. `--pack` claws some of that back, at the cost of a second rounding llama.cpp never pays.
- On the numpy backend it decodes weights to F32, so it holds **twice** the F16 file, where llama.cpp memory-maps the file as-is. (The MLX backend keeps float16 and does not, which is most of why it is faster.)
- `--stream` costs about 1.2 seconds per gigabyte of model, per token, because every token re-reads and re-converts everything. Measured: 0.26 s/token for a 0.25 GB model, 2.74 s/token for a 2.31 GB one. A 70B would be around three minutes per token.

So `--stream` is a demonstration that the weights really are data paged in on demand — not a way to run a model you could not otherwise run. **It will not make a large model fit on a small device.** llama.cpp already memory-maps its files, so it runs models larger than RAM today, and the OS page cache does that better than anything here.

Where the small-machine story is real is *upstream* of inference: a 70B model can be merged, diffed, and packaged on a laptop that could never run it — then exported to GGUF and served by llama.cpp.

## What you can use it for

| You want to | Command | What you get |
|---|---|---|
| Ship a fine-tune without shipping a model | `reminis diff` → `reminis apply` | A delta pack that rebuilds the fine-tune from the base, hash-verified. Often 0.4–12% of the model. |
| Distribute a LoRA to people who don't use peft | `reminis lora` | The adapter as an exact delta pack, byte-identical to peft's own merge. |
| Keep a base and everything derived from it | `reminis registry` | One file. Bases stored outright, derived models as deltas. |
| Know what a fine-tune actually changed | `reminis diff` | Per-tensor change counts, L2 norms, max deltas — a report, not a hash mismatch. |
| Find the step where training went wrong | `reminis log`, `reminis rollback` | Loss spikes and per-parameter gradient norms as SQL, then a hash-verified rewind to a snapshot. |
| Combine two fine-tunes | `reminis merge` | Linear, slerp, task arithmetic, or TIES, with provenance recorded in the result. |
| Subtract a capability | `reminis merge --scale -1` | The base with a fine-tune's task vector removed. |
| Understand a model you were handed | `reminis view`, `reminis info` | Architecture diagram, MoE routing, per-tensor statistics, in a browser. |
| Ask questions about weights | any SQL client | The tensors are rows. Sort by magnitude, group by layer, join across models. |
| Check a model still works after surgery | `reminis run` | It generates text, or it does not. |
| Move between GGUF and safetensors | `reminis convert`, `reminis export` | Lossless in both directions, verified by SHA256. |
| Run a quantized model you were given | `reminis run` | Every K-quant and i-quant, unpacked at load; `--pack` to keep it small. |

## How it compares

reminis overlaps with several tools and replaces none of them. Being specific about that is more useful than a feature table.

**llama.cpp / GGUF** — a runtime, and the one you should serve with. There is no competition here: reminis reads GGUF, gives you things to do with it, and writes it back out. `reminis run` exists to verify a database, not to replace a runtime. On quantized weights llama.cpp is ahead — measured at 86% of it on Mistral-7B Q4_K_M — and the one place reminis leads is running an f16 file without converting it first.

**safetensors** — a storage format, and a good one: memory-mapped, safe to load, fast. It stores *a* model. reminis stores models *and their relationships* — which one came from which, what changed, what it costs to store the difference. reminis reads and writes safetensors, including BF16 and sharded checkpoints.

**Hugging Face Hub and git-lfs** — how models are versioned today, and where reminis has its clearest advantage. git-lfs stores each version as a whole blob, so a base plus twenty fine-tunes is twenty-one full copies. reminis stores derived models as verified deltas, and a fine-tune's delta is typically a small fraction of the model. This is the "40 years of database engineering" claim actually cashing out: the file format understands that two models are related.

**mergekit** — the mature model-merging tool, with far more methods than the four here, and it already does out-of-core merging with lazy tensor loading, so reminis holds no memory advantage over it. Merge in reminis is not a better merger; it is merging that falls out of the storage model, aligned by a SQL join, with provenance written into the result and GGUF as a first-class input.

**What is genuinely unlike the alternatives:** weights as queryable rows; diffs and delta packs with hash-verified apply; many related models in one file stored as deltas; training runs recorded as an edit log you can query; and being able to ask a stored model to generate text as a correctness check. The last one is a small feature that makes the rest trustworthy.

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

### Inference speed, against llama.cpp

`reminis run` is numpy. It exists to show the database is a working model, not to be a fast runtime, so the honest thing is to measure it beside the tool people actually use.

Apple M5, same F16 files for both, `llama-bench` for llama.cpp. The two were run **interleaved** — one round of each, alternating, medians reported — because the machine was not idle, and alternating makes shared load cancel instead of landing on whichever ran second. Prompt processing (`pp512`) and token generation (`tg128`), in tokens per second:

| Model | reminis (numpy) | llama.cpp, CPU | llama.cpp, Metal |
|---|---|---|---|
| SmolLM-135M f16 | 946 pp / **86** tg | 540 pp / **97** tg | 18,126 pp / **176** tg |
| Qwen2.5-0.5B f16 | 502 pp / **27** tg | 330 pp / **30** tg | 5,272 pp / **69** tg |
| Llama-3.2-1B f16 | 209 pp / **11** tg | 153 pp / **13** tg | 1,652 pp / **30** tg |

Against llama.cpp **on the CPU**, numpy generates at **0.86–0.89×** its speed — strikingly consistent across three model sizes — and processes prompts 1.4–1.8× faster. Both end up in Apple's Accelerate for the large matrix multiplies, and prompt processing is one big GEMM where that dominates. This was not the expected result: the first version of this section asserted an order of magnitude, from reasoning rather than measurement, and was wrong.

Against llama.cpp **on the GPU**, it is 2.6–2.8× slower at generation and 8–19× slower at prompt processing. That is the comparison that matters for choosing a runtime, and numpy has no answer to it.

Two caveats point the other way, and neither is fixable by tuning:

- reminis decodes every weight to F32 and keeps it there, so it holds **twice the F16 file** in memory where llama.cpp memory-maps the file as-is. Token generation is bandwidth-bound, so that accounts for much of the remaining gap by itself.
- The comparison is confined to F16 for a like-for-like matmul. reminis does run quantized models now — see below — but by unpacking them, where llama.cpp multiplies the original blocks directly and pays no second rounding.

Then there is `--stream`, which is a demonstration rather than a capability:

| Model | Database | `--stream` |
|---|---|---|
| SmolLM-135M f16 | 0.25 GB | 0.26 s/token |
| Llama-3.2-1B f16 | 2.31 GB | 2.74 s/token |

Nothing is cached, so every token re-reads and re-converts the entire model — eight tokens of SmolLM read **2,796 MB across 2,457 queries** out of a 258 MB file. Peak memory is one layer instead of one model, which sounds like it should let a large model run on a small machine. It does not. The cost is flatly linear in model size, about 1.2 seconds per gigabyte per token, so a 7B lands near 17 s/token and a 70B near three minutes. `--stream` shows that the weights are genuinely data paged in on demand; it is not a way to run a model that would not otherwise fit, and llama.cpp already memory-maps its files for exactly that case.

### Quantized models

Quantized tensors are unpacked at load through the `gguf` package, which reminis already depends on for reading the format. Every quantization llama.cpp writes works, including the i-quants: **Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ3_M, IQ4_XS** all generate coherent text.

The check that matters is not that it generates — wrong block arithmetic still produces fluent nonsense — but that the unpacked weights match the floats they were quantized from. The same SmolLM-135M exists here in both Q4_K_M and F16, so that is a direct comparison: **all 272 tensors correlate at 0.99728 or better**, with relative errors of 0.7–4.3%, which is exactly what Q4_K/Q5_0/Q6_K/Q8_0 rounding looks like. Wrong unpacking would give correlation near zero.

**Be clear what this is not.** Unpacked blocks become float16 in memory, so the Q4_K_M model holds 772 MB against the F16 model's 784 MB — near-identical, from a file that is 101 MB rather than 258 MB. **Quantization saves disk and download here, not RAM.**

`--pack` keeps them packed instead. Which mode you want depends on how the model is stored:

| Mode | What it does | Works on |
|---|---|---|
| `--pack` | Moves GGML's own blocks into the backend's layout, **bit-exactly** | quantized models only |
| `--pack compact` | The same with float16 scales — 17% smaller, 8e-04 error | quantized models only |
| `--pack 4/6/8` | Re-quantizes to that width | any model, float or quantized |

The first two only *rearrange* existing quantization, so they have nothing to do on a model stored as floats — `reminis run` says so rather than leaving you to wonder why the flag changed nothing. For an F16 file, `--pack 8` is the one that helps, and it is imperceptible: correlation 0.9996–0.9999 against the unpacked model.

**`--pack` with no value moves GGML's own blocks into the backend's layout, bit-exactly.** This is the interesting one. Several GGML quantizations turn out to be *already affine* within each group of 32 weights — `Q4_K` is `(d·sc)·q − (dmin·m)`, `Q5_0` is `d·(q−16)`, `Q8_0` is `d·q` — which is precisely the `scale·q + bias` form MLX's quantized matmul expects. So they can be rewritten into its layout by shuffling bits, with **no arithmetic on any weight and no second rounding**. Verified against the `gguf` package's own dequantization: Q4_0, Q4_1, Q5_0, Q5_1, Q5_K, Q4_K and Q8_0 all come back **bit-identical**, zero difference.

**`--pack 4|6|8` re-quantizes instead**, which is smaller and faster but rounds every weight a second time.

| | Weights resident | Logits vs unpacked |
|---|---|---|
| unpacked (default) | 258 MB | — |
| `--pack` (bit-exact) | 145 MB | corr 0.9998979, top-5 intact |
| `--pack compact` | 138 MB | corr 0.9998995, top-5 intact |
| `--pack 4` | 122 MB | corr 0.9711, top-5 **reordered** |

`--pack compact` holds the per-group scale and bias in float16 instead of float32. At a group size of 32 that is the difference between 6 and 5 bits per weight on a 4-bit tensor — 17% of the model — for a measured **8.3e-04** relative error, far below the quantization already in the file. It is the one thing that stops the repack being bit-exact, so it is opt-in.

`Q6_K`, `Q2_K`, `Q3_K` and the i-quants have no affine form — 16-weight sub-blocks and codebooks respectively. Rather than leave them as float16, which would cost *more* memory than the file did, they are re-quantized to the nearest width. On Mistral-7B that one change was worth 6.55 GB → 5.49 GB and 3.0 → 8.3 tok/s.

### Mixture-of-experts models

The experts are stored stacked — one 3-D tensor per projection with the expert as its first axis — so a router's top-k choice becomes a gather, and on MLX a single `gather_qmm` call selects the experts and multiplies by them in one kernel. They pack like any other weight, which matters more here than anywhere else: on Granite-3.1-1b-a400m the experts are 1.2B of the 1.33B parameters.

| | Weights resident | Generation |
|---|---|---|
| unpacked | 2.50 GB | 38.7 tok/s |
| **`--pack compact`** | **0.90 GB** | **121.9 tok/s** |
| llama.cpp Metal | 0.78 GB | 139.7 tok/s |

Packing the experts is worth **3.2×**, and lands at 87% of llama.cpp from a file of 0.78 GB. Both backends produce identical text, which is the check that matters — routing to the wrong experts, weighting them wrong, or dropping one of Granite's four scaling multipliers all produce fluent nonsense rather than an error.

### A 7B on a 16 GB laptop

The largest thing that fits here, measured rather than estimated. Mistral-7B-Instruct v0.3 at Q4_K_M, a 4.07 GB file, against llama.cpp on the same file and machine:

| | Weights resident | Generation | vs llama.cpp |
|---|---|---|---|
| llama.cpp Metal (`llama-bench`, tg64, r=3) | 4.07 GB | 28.4 ± 0.1 tok/s | — |
| **`--pack compact`** | **4.80 GB** | **24.3 tok/s** (25.0 / 23.8 / 24.0) | **86%** |

An earlier version of this table said 13.1 against 12.7 tok/s, and read that as reminis being 3% ahead. That is no longer true and was never as solid as it looked. Re-measuring both sides on the current homebrew build (ggml 0.18.1), llama.cpp went from 12.7 to 28.4 tok/s and reminis from 13.1 to 24.3 — both roughly doubled, and llama.cpp doubled harder. **reminis is at 86% of llama.cpp here, not 103%.** The lesson is about method rather than about either program: a ratio between two numbers measured months apart, against a dependency that ships optimizations continuously, is not a measurement.

Everything above 7B on this machine is a question of arithmetic rather than capability: a 13B at Q4_K_M would land near 8.8 GB packed compact, which fits; a 30B would not.

Mistral-7B runs end to end now that SentencePiece is implemented:

```
$ reminis run models/mistral7b.db "The capital of France is" -n 24 --temp 0 --pack compact
The capital of France is Paris, but the largest city is Marseille.
```

One caveat on that row: loading takes ~83 s, because the repack runs in numpy. It is one pass over the model and nothing has been done to speed it up.

This closes most of the gap with llama.cpp on quantized weights — the values multiplied are now exactly the ones in the file — without closing it entirely: llama.cpp still reads the original blocks with no repack step and no float32 scale array beside them.

Two other things came out of re-measuring. `PRAGMA mmap_size` is worth 7% here (25.1 against 23.5 tok/s), where on a database bigger than the machine it costs 3.7×, so reminis now maps a file only when it is at most half of physical memory — 4.4 GB and 22.9 GB being the two points either side of that line. And the 41 s prefill in these runs is the numpy repack, still unimproved, still a one-off.

### A 20B model in 3.65 GB, by only reading the experts it uses

gpt-oss-20b is 11.27 GB and will not fit alongside anything else on a 16 GB machine. But **84% of it is experts, and a token routes to 4 of 32** — so loading all of them to use an eighth is the waste, and the router says which are wanted *before* any are read.

`--expert-cache` reads each chosen expert as a byte range and keeps the recently used ones:

| | |
|---|---|
| Model on disk | 11.27 GB |
| Read per token | 1.18 GB, not 9.46 |
| **Resident** | **3.65 GB** |
| Generation | 0.71 tok/s |

Slow, and it runs at all — which it does not otherwise. This is the one thing the database buys that a memory-mapped file cannot: a deliberate read of the rows the routing selected, rather than the OS guessing from access patterns after the fact.

Two things made it usable rather than merely possible. Profiling showed the database read was **1.1 ms** and irrelevant; the cost was 54 ms unpacking MXFP4 in numpy and 38 ms uploading a 33 MB float32 expansion of a 4.2 MB block. Unpacking on the GPU instead uploads the bytes the file actually holds — **12.5× faster, and bit-identical** to numpy's. That took 0.05 tok/s to 0.71.

Incremental blob reads arrived in Python 3.11, so this needs 3.11 or newer. On 3.10 the whole expert stack is read at once, and `reminis run` says so.

### 0.71 to 37 tok/s, by building an index over the experts

0.71 tok/s runs but is not usable. Profiling one token found that 98% of it was the mixture-of-experts, and 60% of *that* was not reading weights. Fetching one expert cost 8.65 ms: **2.5 ms of read and 6.1 ms of arithmetic** — unpacking MXFP4's 17-byte blocks and re-packing them into the layout the matmul kernel wants, for ~116 experts per token, and again next token for the same experts because the cache had evicted them.

That arithmetic is a pure function of bytes already in the database, so it belongs in the database:

```bash
reminis prepare model.db          # build it
reminis prepare model.db --drop   # throw it away, reclaim the space
```

`prepare` writes a second physical copy of the expert weights — already unpacked, already in the kernel's layout, one row per expert, clustered so a layer's experts are contiguous. It is an index in the ordinary sense: redundant, derived, ordered for one access path, and droppable without losing anything. On gpt-oss-20b it is 2,304 rows and takes 30 seconds.

Then `--experts all` holds the whole index and pins it:

```bash
reminis run model.db "Why is the sea salty?" --pack 4 --experts all
```

| | Baseline | Indexed | |
|---|---|---|---|
| Expert fetch | 8.65 ms | 2.6 ms | the 6.1 ms is paid once, at build |
| Resident | 3.65 GB | 8.4 GB | at `--bits 3`, on a 16 GB machine |
| **Generation** | **0.71 tok/s** | **37.3 tok/s** | 53× |

Four things had to be true at once, and leaving out any one gave back most of it:

**Stop memory-mapping the file.** `PRAGMA mmap_size` was measured on a 258 MB model, where mapping everything is free. It stops being free when the file is bigger than the machine: touched pages stay resident and compete with the weights. Measured **1.70 ms per block mapped against 0.46 ms unmapped** — 3.7× the wrong way. reminis now maps the file only when it comfortably fits.

**Don't thread the reads.** A pool of readers overlapping I/O with compute made it *slower* (0.54 ms/block against 0.46 serial). The read was never the thing waiting.

**Touch the weights, not just load them.** A buffer copied to the device has not been *read* by it, and the first read faults it in. A token whose experts had never been multiplied took 96 ms against 9.5 ms once they had — so the rate climbed from 1.4 to 24 tok/s over seventy tokens and never reached its ceiling. Multiplying each expert by a zero vector at load costs a few seconds and moves the whole curve to the front.

**Pin the memory.** Unified memory is ordinary system memory, and macOS compresses it under pressure — silently, after which the GPU stalls faulting it back. Left to the system, the rate wandered between **4.6 and 33 tok/s** between ten-token windows. Wired, it was a flat **41 tok/s**.

The honest limits. A bigger cache is not better: at 1,400 experts it was *slower* than at 400, because the decoded cache and the OS page cache compete for the same 16 GB. And the index has to fit — a 4-bit index is 10.75 GB against this device's 12.7 GB working set, and overcommitting it produced fluent-looking nonsense rather than an error, so `--experts all` now measures the fit first and refuses with the two ways out. 3 bits fits and costs quality; 4 bits needs a machine with more memory.

The thing that made this possible is the same thing the whole project is about. The expensive step was a deterministic function of stored bytes, and a database is where derived representations of stored bytes belong. A GGUF file has nowhere to put one.

**There is no llama.cpp figure in that table, and it is not an oversight.** llama.cpp would not run gpt-oss-20b on this machine at all. Every configuration tried — full offload, `-ngl 20`, `-ngl 16`, and `-ncmoe 12`, which exists for exactly this case — failed the same way:

```
test_prompt: failed to decode prompt batch, res = -3
```

The cause is the wall reminis met too: `recommendedMaxWorkingSetSize = 12713 MB` against an 11.3 GB model, leaving nothing for the cache and compute buffers. It may also be a version problem rather than a memory one — this is the homebrew build, and gpt-oss needs recent MXFP4 and attention-sink support — which has not been chased down.

So the claim here is about **fitting, not speed**: on a 16 GB machine, llama.cpp did not run this model in the configurations tried and reminis did. Reading it as "reminis is faster than llama.cpp on gpt-oss" would be wrong twice over, because the 37.3 tok/s is a 3-bit index against llama.cpp's MXFP4 and those are not the same model. Where the two can be compared precision-matched, on Mistral-7B above, llama.cpp is ahead.

### Compressing the key/value cache

The weights are a fixed cost; the cache grows with every token. At long context it is the cache, not the model, that decides whether a prompt fits — so `--kv-bits` compresses it.

Measured on a 1536-token context:

| | Cache | Generation | vs uncompressed |
|---|---|---|---|
| off | 39.2 MB | 236 tok/s | — |
| `--kv-bits 8` | 21.6 MB (1.8×) | 176 tok/s | identical text, correlation 0.99999 |
| `--kv-bits 4` | 12.3 MB (3.2×) | 188 tok/s | correlation 0.9946 |

**This costs speed rather than saving it**, which is the opposite of `--pack` and worth being explicit about. MLX has no quantized attention kernel, so the cache is decompressed on every layer of every token; what you buy is room, not time. Leave it off unless the cache is what is stopping you.

llama.cpp has the same feature as `-ctk`/`-ctv`, and unlike its weight quantization that one *is* a runtime flag — this is the one place the two tools line up directly.

### Backends: numpy, MLX, CuPy

`reminis run` computes through whichever array library suits the machine — **MLX** on Apple silicon, **CuPy** on NVIDIA, **numpy** everywhere. numpy stays the reference implementation: it is the one whose logits are checked against `transformers`, and the others earn their place by agreeing with it.

Same models, same machine, `--backend numpy` against `--backend mlx`:

| Model | numpy | MLX | Speedup |
|---|---|---|---|
| SmolLM-135M f16 | 975 pp / 86 tg | **26,359 pp / 194 tg** | 27× pp / 2.3× tg |
| Qwen2.5-0.5B f16 | 522 pp / 27 tg | **8,775 pp / 70 tg** | 17× pp / 2.6× tg |
| Llama-3.2-1B f16 | 274 pp / 11 tg | **1,962 pp / 32 tg** | 11× pp / 2.9× tg |

The gain comes less from the GPU than from float16 being a native compute type there: MLX never pays the widening that costs numpy 213 ms of a 274 ms load, and it holds half the memory as a result.

### Beating llama.cpp by packing on the way in

An f16 model can be packed as it loads, with no separate quantization step and no second file. On the same f16 GGUF llama.cpp is reading, interleaved and best-of-four:

| Model | reminis f16 | reminis `--pack 8` | llama.cpp f16 | packed vs llama.cpp |
|---|---|---|---|---|
| SmolLM-135M | 274 | 349 | 325 ± 19 | **107%** |
| Qwen2.5-0.5B | 108 | 173 | 114 ± 8 | **152%** |
| Llama-3.2-1B | 49 | 91 | 54 ± 0.3 | **170%** |

These were re-measured against the current llama.cpp alongside the 7B above, and unlike that one they held: llama.cpp's f16 rates barely moved (337 → 325, 123 → 114, 52 → 54), because the work that has gone into it since is in the quantized kernels rather than the f16 path. That is the same reason the Q4_K_M comparison flipped and this one did not.

`--pack 8` correlates with the unpacked model at **0.9996–0.9999** and produces identical greedy text, so this is not a quality trade in any sense that shows up in output — it is the same model reading half the bytes.

Three things make it work:

- **The embedding table is packed too.** It is normally excluded because it is indexed rather than multiplied — but a packed table can still be indexed, since the rows are contiguous, and on a tied-weights model it is also the output projection, where it is the single largest read of every token.
- **The per-group scales are held in float16** rather than float32, which is what `compact` does: 5 bits per weight instead of 6 on a 4-bit tensor.
- **The group size is chosen per tensor**, largest that divides the row. A bigger group carries fewer scales, so it is smaller *and* faster: on Qwen at 8 bits, a group of 128 gives 174 tok/s and 625 MB against 165 and 682 for a group of 32, for a correlation that falls only from 0.99989 to 0.99972. It has to divide the row length, which is why it is per tensor rather than global — 896 takes 128, 576 does not and takes 64. That one change was worth 74 → 87 tok/s on Llama-1B.

Be clear about the boundary. Precision-matched, llama.cpp is still ahead: its **Q8_0** kernels beat this at 395 ± 20 against 349 on SmolLM, and f16 against f16 it wins everywhere by 7–20%. The claim is narrower and still useful — *given an f16 file and no willingness to convert it*, reminis will run it faster than llama.cpp will.

### Against llama.cpp, generating tokens

Token generation was the one axis llama.cpp clearly won. These are run **interleaved** — one round of each, alternating, best of seven — because the machine is not idle and alternating makes shared load cancel:

| Model | reminis (MLX) | llama.cpp Metal | Ratio |
|---|---|---|---|
| SmolLM-135M | 194 tg | 261 tg | 74% |
| Qwen2.5-0.5B | 70 tg | 78 tg | 90% |
| **Llama-3.2-1B** | **32 tg** | **32 tg** | **101%** |

**At 1B the numpy-and-MLX forward pass matches llama.cpp**, and the ratio improves monotonically with model size — because what remains is a fixed per-token cost that larger matrices amortise. Prompt processing was already ahead and is now 26,359 against 18,126 on SmolLM.

Four things got it there, in order of size:

- **`mx.fast.scaled_dot_product_attention`.** The hand-rolled version materialised a scores matrix and expressed grouped-query attention with a broadcast axis. The fused kernel does neither and handles GQA internally.
- **`mx.fast.rope`.** Rotary embedding was six array operations per projection per layer; it is now one.
- **Weight lookups hoisted out of the loop.** Each layer rebuilt eight weight-name strings and hashed eight dictionary keys, every token — 360 string operations per token on a 30-layer model, which is real time when a token takes under six milliseconds.
- **`mx.compile` on the feed-forward half.** Every layer has identical shapes there, so one compiled graph serves all thirty; the weights are traced arguments rather than baked constants.

What is left is dispatch overhead. Profiling puts it at 26% of a token for SmolLM and 18% for Qwen, and removing it entirely would give 238 tok/s and 84 tok/s respectively — ahead of llama.cpp in both cases. That is the remaining work, and it is Python overhead rather than anything about the database.

**On which database ideas helped: none of them, and that is the honest finding.** Memory-mapping the file was worth a real 4.1 → 6.7 GB/s on reads, clustered ordering and prefetching would help `--stream`, and materialising a pre-converted form would cut load time. But once the weights are resident, SQLite is not in the critical path at all — a forward pass touches no rows. Database techniques can improve how fast a model is *loaded* and how cheaply it is *stored*; they cannot make a matrix multiply faster.

Backends are picked per *workload*, not per machine, because a GPU is not a blanket improvement:

| Workload | Backend | Why |
|---|---|---|
| `inference` | MLX / CuPy / numpy | Large matrix multiplies; 7–21× on prompt processing |
| `elementwise` (merge) | numpy | Measured; see below |
| `bytes` (diff) | numpy | On a 4M-element block the XOR is 0.4 ms and zlib is 252 ms. zlib runs on the CPU. |

**The merge row is a lesson worth recording.** A benchmark of one 4M-element block said the GPU was 6.6× faster and bit-identical, because numpy's float16 decode and encode run at ~1.2 GB/s. Wired into the real merge it came out **2.4× slower**. The block was warm and reused; a merge reads each blob out of SQLite once, cold, and pays device-transfer costs the benchmark never exercised. The selection table now carries the measured answer rather than the plausible one.

Half precision moves a logit by a few hundredths — enough to reorder near-tied tokens in a top-5 list, never enough to change the argmax. Across three architectures the backends agree on the next token with correlation ≥ 0.999993, and greedy generation produces identical text. Where exactness matters, `--backend numpy` is always there.

### What made it faster, and what didn't

Profiling put 65% of decode time in the linear layers, so the changes went there. What worked:

- **Stacking Q, K and V into one matrix per layer**, and gate with up — seven matrix products per layer become four, over contiguous memory. The originals are dropped as the fused copy is built, so it costs no extra RAM. Measured on the real shapes, fused QKV is a third faster than the three separate calls.
- **Broadcasting for grouped-query attention** instead of `np.repeat`, which was copying the entire KV cache once per layer per token.
- **Preallocating the KV cache.** Growing it with `np.concatenate` recopies everything on every token, turning a linear cost quadratic — invisible until the context is long.
- **`PRAGMA mmap_size`.** Reading every weight out of a 258 MB database goes from 4.1 GB/s to 6.7 GB/s when SQLite maps the file instead of copying each blob through its own buffer.
- Caching the rotary tables, which were recomputed once per layer — thirty times per token — and skipping the causal mask during single-token decode, where it masks nothing.

Together those took SmolLM-135M from 0.71× llama.cpp's CPU speed to 0.89×. The larger models barely moved, because they are already bandwidth-bound rather than overhead-bound.

Three ideas that seemed obvious and lost, all measured:

- **Threading the matrix products** made it *slower*. Accelerate already reaches 70 GB/s single-threaded; splitting the work across 8 threads dropped it to 17 GB/s.
- **Keeping weights in F16** to halve memory traffic was 34× slower, and the reason is worth stating plainly, because "why convert at all?" is the obvious question. BLAS provides `sgemm` and `dgemm` — single and double precision — and no half-precision equivalent, so numpy has no BLAS-backed matmul for F16 and falls back to a generic loop. On a 4096×4096 matrix-vector product: **0.92 ms in F32, 48.6 ms in F16**, a 53× penalty for halving the bytes. llama.cpp does not convert because it ships its own F16 SIMD kernels and Metal shaders; numpy has neither, so widening to F32 once at load is the cheaper of the two bad options.
- **Hand-written bit-twiddling** for that conversion was slower than numpy's, and wrong on subnormals.

The remaining ceiling is DRAM bandwidth on F32 weights, and numpy has no way to read fewer bytes. One database-shaped answer is left on the table: the F16→F32 conversion costs 213 ms of a 274 ms load, so storing the F32 form alongside the F16 — a materialized view of the weights — cuts load time and makes `--stream` about 1.7× faster, for double the disk. That is on the roadmap rather than in the code.

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

Encoding is **XOR**, not arithmetic subtraction. An arithmetic float delta is not exactly reversible — `b - a` generally is not representable in the tensor's own dtype, so `a + delta` lands a rounding step away from `b`. XOR is exact for every dtype, and also works on quantized tensors, whose bytes cannot be subtracted meaningfully at all. Per tensor, the smallest of a compressed XOR delta, a compressed full replacement, and a bit-plane split of the XOR delta wins.

### How small are packs, really?

It depends entirely on how much of the model the fine-tune touched.

| Scenario | Tensors changed | Pack size |
|---|---|---|
| Targeted change (5 of 272 tensors) | 5 | **1.2%** of full model |
| Full fine-tune (SmolLM-135M to its Instruct variant) | 272 of 272 | **50.4%** of full model |

The second row is the honest one for full fine-tuning. Every tensor changed, with ~97% of individual values differing in each.

### Bit-plane splitting, and where the floor actually is

A 16-bit float interleaves a highly predictable exponent with mantissa bits a fine-tune randomises. Handed the delta as one stream, a general-purpose compressor sees every predictable exponent run chopped up by noise every other byte. Splitting the delta into two streams — one per byte position — and compressing each separately took the full fine-tune above from **58.4% to 50.4%**.

It is picked per tensor by the same smallest-wins rule as everything else, so it can only ever shrink a pack; a tensor that gains nothing keeps an encoding that older readers understand.

That was worth doing, but the more useful result was finding out how much room is left. Measuring the delta field by field on those 134.5 M weights:

| Field | Behaviour under a full fine-tune |
|---|---|
| Sign bit | flips in 1.7% of weights |
| Exponent | unchanged in 82.8%, within ±1 in 97.1% |
| Mantissa bits 3–6 | flip in **50.0%** — indistinguishable from coin tosses |
| Mantissa bits 0–2 | flip in 0.01% (these weights are bf16 precision widened into f16) |

Four bits per weight that flip at exactly 50% are incompressible by definition, and four bits of sixteen is 25% of the file on their own. Adding the exponent and the upper mantissa bits, the order-0 entropy of the delta is **46.5%** of the raw weights. The shipped encoder is at 50.4%, within four points of a floor no lossless coder can pass.

So the target of getting full-fine-tune packs to ~25% ([#7](https://github.com/ronxldwilson/reminis/issues/7)) is **not reachable losslessly**, and this documents why rather than leaving it open as though better compression would get there. Packs below that floor have to give up exactness, which is what `--lossy` already does — and for a full fine-tune it declines, because the delta is not low-rank. The place delta packs win big is LoRA-shaped updates, below, where they hit 0.3–1.4%.

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
| SmolLM-135M, 258 MB | 35.1 MB (13.7%) | **3.5 MB (1.4%)** | 10.1x | 1.2e-04 |
| Llama-3.2-1B, 2.4 GB | 232.1 MB (9.4%) | **6.7 MB (0.3%)** | 34.6x | 1.1e-04 |

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

### Merging models that do not fit in memory

Tensors are combined in **row-blocks**, so peak memory is set by the block size rather than by the largest tensor in the model. Nothing ever holds a decoded copy of a whole tensor — which matters because a large model's embedding matrix is the single thing that would otherwise decide the ceiling.

Merging a 2.31 GB Llama-3.2-1B, whose embedding matrix alone is 501 MB:

| | Peak memory | Time |
|---|---|---|
| Whole tensors at a time | 5,951 MB | 15.1s |
| Row-blocks, Python 3.10 | 2,162 MB | 13.4s |
| Row-blocks, Python 3.11+ | **185 MB** | 9.7s |

Python 3.11 added incremental blob I/O, which reads a byte range out of a BLOB without materialising the rest, and writes one back the same way. On 3.11+ nothing model-sized is ever resident, so the peak is flat in model size: merging a 70B checkpoint costs the same 185 MB as merging this 1B one. On 3.10 the blob is fetched whole and sliced, which still avoids decoding it to float32 but keeps the stored bytes around.

To be clear about what this is and is not worth: [mergekit](https://github.com/arcee-ai/mergekit) already merges out-of-core with lazy tensor loading and runs on a laptop too, so this is not an advantage over it. What it does mean is that merging in reminis costs no more memory than the storage layer already did, and that a model far too large to run can still be merged, diffed and packaged on the machine in front of you.

(SQLite's own `substr` looks like a third option and is not: it materialises the entire blob to answer each call, so reading a 501 MB tensor in blocks would read it once per block.)

Two of the four methods need a quantity measured over the whole tensor before any block can be combined, and both get their own pass:

- **slerp** needs the angle between the two models, which is two norms and a dot product — the angle between two vectors is not the angle between their first thousand entries.
- **ties** needs the magnitude threshold that trimming keeps, which is a rank over every entry. Blocked, that is a histogram narrowed until the band containing the threshold is small enough to rank exactly. The result is the number `np.partition` would have returned, not an approximation of it — tested against it directly, including on a vector where 95% of entries share one magnitude.

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

# Pick the array library by hand; auto uses the fastest one available
reminis run model.db "The capital of France is" --backend numpy
```

`--stream` holds nothing in memory between uses, so peak memory is one layer rather than one model, and a 258 MB database serves 2,796 MB of reads across 2,457 queries to generate eight tokens. It is the thesis in one flag — the model is data, paged in on demand — and it is a demonstration, not a deployment mode. It costs about 1.2 seconds per gigabyte of model per token, so it does not let a large model run on a small machine, and [it is not trying to](#what-this-is-and-what-it-is-not).

**What `reminis run` is for** is the loop it closes. A merged, rolled-back, or delta-applied database can be checked by asking it to speak, which no hash can do for you:

```bash
reminis merge base.db instruct.db -o soup.db
reminis run soup.db "Name three colours." --chat --temp 0
```

Implemented, and enforced rather than assumed:

- **llama-family, qwen2 and Granite architectures** — RMSNorm, SwiGLU, grouped-query attention, rotary embeddings. That covers llama, llama 3 (including its per-dimension rotary scaling, which is stored as a tensor rather than as metadata), mistral, qwen2 (which uses the other rotary layout and has QKV biases), smollm, and Granite (which scales embeddings, residuals, logits and attention by its own constants — `attention.scale` is 1/head_dim where everyone else uses 1/sqrt(head_dim)).
- **Mixture of experts** — a router picks the top-k of many stacked feed-forward networks per token. The experts are one 3-D tensor per projection, so selecting them is a gather rather than a Python loop, and they pack like any other weight.
- **gpt-oss**, which needs all of the above plus YaRN rotary scaling, per-expert biases, and its own gated activation — it clamps both projections, uses `sigmoid(1.702·gate)`, and multiplies by `(up + 1)` rather than `up`.
- **Attention sinks** — a learned per-head logit that joins the softmax denominator without contributing a value, letting a head attend to nothing in particular.
- **Sliding-window attention** — layers that see only the most recent N keys, alternating with full-attention layers on whatever pattern the metadata records.
- **Float weights** — F32, F16, BF16.
- **Two tokenizer families**, both rebuilt from the database. Byte-level BPE (`gpt2`) matches `transformers` exactly across three vocabularies. SentencePiece (`llama`) matches llama.cpp exactly on 14 strings including special tokens, byte fallback and whitespace edges — it has no merge list at all, merging instead by a score attached to each token, so the vocabulary itself encodes the merge order.

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

## Tests

Fourteen suites, 338 explicit checks, run with `uv run python tests/<name>.py`. They need no framework and skip rather than fail when a model or an optional dependency is absent.

The references they check against — transformers for logits, peft for LoRA adapters, torch, and MLX on Apple silicon — are declared as a dev dependency group, so `uv sync` installs them and does not remove them. That is worth stating because getting it wrong is quiet: when those packages went missing, the three suites that compare reminis against a reference implementation degraded to skips and went on reporting PASS.

| Suite | Covers |
|---|---|
| `test_roundtrip` | SHA256-verified GGUF round-trip across 20 models and 13 quantization types |
| `test_diff`, `test_lowrank` | Delta packs, hash-verified apply, low-rank encoding |
| `test_bitplane` | Bit-plane delta encoding: reversibility, refusals, older packs still read |
| `test_safetensors`, `test_lora_adapter` | Safetensors both directions, peft agreement |
| `test_track`, `test_registry` | Training logs, rollback, many models in one file |
| `test_viewer` | The architecture diagram, by running the page's own JS under node |
| `test_merge` | Four merge methods, chunking against whole-tensor results, refusals |
| `test_infer` | Tokenizers against `transformers` and llama.cpp, logits against torch, KV cache, MoE, quantization |
| `test_backend` | numpy/MLX/CuPy agreement on primitives, attention, sinks, windows |
| `test_ggml_affine` | Bit-exactness of every repackable quantization |
| `test_features` | Feature combinations and the command line |

The checks are written to fail for the right reason. Where a property could pass vacuously — two backends agreeing because both ignore a feature, a cache "matching" itself — the test compares against an independent reference instead: `transformers` for logits, `llama-tokenize` for token ids, `gguf` for dequantization, `peft` for LoRA merges. Several were verified by deliberately breaking the code and confirming they caught it.

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
- [x] GPU backends chosen per workload (`--backend`: MLX on Apple silicon, CuPy on NVIDIA)
- [x] Better delta encoding: bit-plane splitting, 58.4% to 50.4% — and a measured entropy floor of 46.5% showing how little is left ([#7](https://github.com/ronxldwilson/reminis/issues/7))
- [ ] Storing the F32 form beside the F16 one, so loading skips the conversion
- [x] Running quantized models: every K-quant and i-quant, unpacked at load
- [x] Keeping weights packed in the backend's own format (`--pack 4/6/8`)
- [x] Multiplying GGML's blocks with no second rounding, for the affine types (`--pack`)
- [x] The remaining quant types re-quantized to the nearest width rather than left as float16
- [x] Mixture-of-experts models: router, stacked experts, packed and gathered
- [x] SentencePiece tokenizers, verified against llama.cpp token for token
- [x] Key/value cache compression (`--kv-bits`), for when the context is the constraint
- [x] A materialized index over MoE experts (`reminis prepare`): gpt-oss-20b from 0.71 to 37 tok/s
- [ ] Faster repacking at load: 83 s for a 7B, all of it numpy
- [ ] Running attention-free architectures (Mamba / state space, RWKV)
- [ ] Unsloth integration

## License

MIT
