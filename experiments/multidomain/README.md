# Ten fine-tunes, one base, one file

Takes Qwen2.5-0.5B-Instruct, trains ten LoRA adapters on ten unrelated corpora,
and measures the two things a delta pack is supposed to buy: what eleven
related models cost to keep, and whether each one actually learned its subject.

## Running it

Four scripts, in order. Each is resumable — re-running skips what already
exists.

```bash
uv run experiments/multidomain/prepare_datasets.py      # ~5 min, downloads 10 datasets
uv run experiments/multidomain/fix_remaining.py         # datasets the first pass missed
uv run experiments/multidomain/finetune_all.py          # ~25 min, 10 LoRA runs
uv run experiments/multidomain/benchmark.py             # fuse, convert, diff, generate
uv run experiments/multidomain/benchmark_perplexity.py  # ~5 min, the real measurement
uv run experiments/multidomain/merge_experiment.py      # ~4 min, ten models into one
uv run experiments/multidomain/specialist_plus.py       # ~6 min, can a merge beat one?
```

`finetune_all.py` draws a live dashboard: per-domain progress, loss, tokens/sec,
peak memory, and an ETA. It checkpoints every 50 iterations and resumes from the
latest one, so an interrupted run costs at most a couple of minutes. Per-domain
logs land in `logs/`.

Needs ~25 GB of disk and about 4 GB of RAM at peak. Everything downloads on
demand; nothing here is checked in but the scripts, the training splits, the
adapter configs, and the two result files.

## What it produces

```
base_model/       Qwen2.5-0.5B-Instruct, downloaded          (999 MB)
datasets/         10 domains x train/valid/test.jsonl        (3 MB, tracked)
adapters/         10 LoRA adapters                           (267 MB)
fused_models/     10 standalone models, adapter merged in    (9.3 GB)
reminis_dbs/      11 databases + 10 delta packs              (11 GB)
merged_dbs/       merge outputs, plus specialist+alpha ones  (15 GB)
merged_models/    the same, exported so mlx_lm can load them (15 GB)
logs/             per-domain training and benchmark logs
```

## The domains

| domain | source |
|---|---|
| code | `iamtarun/python_code_instructions_18k_alpaca` |
| creative | `roneneldan/TinyStories` |
| finance | `gbharti/finance-alpaca` |
| history | `cais/mmlu` (prehistory) |
| legal | `nguha/legalbench` (contract_nli) |
| math | `openai/gsm8k` |
| medical | `medalpaca/medical_meadow_medical_flashcards` |
| reasoning | `allenai/ai2_arc` (ARC-Challenge) |
| science | `allenai/sciq` |
| sql | `b-mc2/sql-create-context` |

500 training examples each where the source had them; legal (89) and history
(304) are smaller because their sources are.

## Results

**Storage**, for a base and ten derived models:

| | total |
|---|---|
| eleven safetensors directories | 11.00 GB |
| eleven reminis databases | 10.99 GB |
| one base database and ten delta packs | **2.12 GB** |

Each fine-tune changed 56 of 290 tensors, so each pack is ~112 MB against a
999 MB model. The middle row is the useful control: eleven databases save
nothing, because a database of one model is the model.

**Capability**, as perplexity on each domain's held-out split:

| model | own domain | base on the same | change |
|---|---|---|---|
| ft_sql | 2.07 | 19.98 | -89.6% |
| ft_legal | 4.74 | 29.69 | -84.0% |
| ft_reasoning | 2.50 | 13.25 | -81.1% |
| ft_history | 3.28 | 15.47 | -78.8% |
| ft_medical | 3.16 | 13.94 | -77.3% |
| ft_science | 4.49 | 17.63 | -74.5% |
| ft_finance | 7.70 | 23.05 | -66.6% |
| ft_code | 2.23 | 6.28 | -64.5% |
| ft_creative | 4.01 | 9.61 | -58.3% |
| ft_math | 1.80 | 4.14 | -56.5% |

Every fine-tune also improved every *other* domain. Base averages 15.3 across
the ten splits; the worst fine-tune averages 6.3 and the best 4.9. At this size
— 200 iterations, eight layers, 500 examples — LoRA is teaching the shape of an
instruction-response pair at least as much as any one subject, and none of the
ten forgot anything measurable.

## Merging the ten into one

`merge_experiment.py`. Base averages 15.30 across the ten splits; the
specialist diagonal — each domain scored by the model trained on it, which is
ten models — averages 3.60. One merged model:

| merge | mean ppl | of the way from base to specialists |
|---|---|---|
| `ties` (8 domains) | 4.47 | 92.5% |
| `linear` (10) | 4.54 | 92.0% |
| `task-arithmetic` (10) scale 0.1 | 4.54 | 92.0% |
| `linear` (8 domains) | 4.56 | 91.8% |
| `task-arithmetic` (10) scale 0.3 | 4.84 | 89.4% |
| `task-arithmetic` (10) scale 1.0 | 113.57 | destroyed |

Its delta against base is 101 MB, so one model competent across all ten costs
1.10 GB against 2.12 GB for the ten specialists kept apart.

Three things, none of them expected. **The method barely matters** — linear,
TIES and task arithmetic at 0.1 are distinct operations that agree to within
0.09 perplexity; TIES costs 50.5s against linear's 4s and wins only on legal
(5.7 against 7.9), the smallest split, where trimming protects a signal a mean
dilutes. **Merging eight models improved the two they never saw** — the
eight-way merges score 5.6 and 3.6 on science and sql against base's 17.6 and
20.0, and adding those domains only moves them to 5.4 and 3.1. **Scale 1.0 is
outside the basin** — ten task vectors at full strength are seven times worse
than not merging.

The ten-way merges compose in two stages because `reminis merge` attaches at
most 8 databases. That is exact for linear and task arithmetic and not for
TIES, so TIES runs natively over eight domains with a matched linear control —
which is what makes science and sql held out rather than dropped.

## Better generalist, worse specialist

Across all ten splits the merge at 4.54 beats every individual fine-tune (best
`ft_finance` 4.85, worst `ft_legal` 6.32). On each domain's own split it loses
to that domain's model ten times out of ten — 6% on finance, 12% on code, 73%
on legal. The gap tracks how idiosyncratic the domain is; legal is 89 examples
with a rigid format, sql is syntax, and an average washes that out.

`specialist_plus.py` tries the repair: keep a domain's task vector whole and
add a fraction of the other nine, `base + tv_d + alpha * sum(others)`. Since
`ta10_s10` is already `base + sum(all ten)`, that is a two-input merge weighted
`[1-alpha, alpha]`.

| domain | base | plain merge | specialist | a=0.05 | a=0.1 | a=0.2 |
|---|---|---|---|---|---|---|
| legal | 29.69 | 8.20 | **4.74** | 4.76 | 4.81 | 5.16 |
| sql | 19.98 | 3.11 | **2.07** | 2.07 | 2.09 | 2.23 |
| code | 6.28 | 2.50 | **2.23** | 2.26 | 2.32 | 2.55 |

Monotone decay, nine losses out of nine. The premise was wrong about where the
shared signal lives — it is in every task vector including the specialist's
own, so `sum(others)` adds nine redundant copies of what the specialist has
plus nine domains' specifics that are noise for this one. What alpha buys is
breadth: at 0.05 the home domain gives up 0-1.3% and the ten-domain mean
improves 4.7-6.2%, in a 123 MB pack rather than 107 MB.

## A negative result about the metric

The first pass scored generated answers by keyword match — does the medical
answer contain "insulin", does the SQL answer contain "GROUP BY". It put the
*base* model on top at 0.88 against a best fine-tune of 0.87, and several
fine-tunes appeared to have catastrophically forgotten other domains.

None of that was true. A keyword score cannot separate "mentions insulin" from
"is right about insulin", and a base instruct model already knows to name the
topic it was asked about. `benchmark.py` still runs that scoring and writes it
to `benchmark_results.json`, because the contrast is the point;
`benchmark_perplexity.py` is the measurement to read.

## Caveats

- 0.5B is small. Whether the no-forgetting result holds at 7B or with longer
  training is a different experiment, which these scripts would run unchanged
  given a different `base_model`.
- Perplexity measures fit to a distribution, not correctness. A model can have
  low perplexity on medical text and still be wrong about medicine.
- The ten test splits come from the same sources as the training splits, so
  these numbers are in-distribution by construction.
