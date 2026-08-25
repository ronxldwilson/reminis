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
logs/             per-domain training logs
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
