"""Can one merged model hold all ten capabilities?

Merges the ten domain fine-tunes with `reminis merge` and measures the result
the same way the individual ones were measured: perplexity on each domain's
held-out split.

`reminis merge` attaches at most 8 models at once, so the ten-way merges are
composed in two stages. That composition is exact for two of the three methods:

  linear           avg(m1..m8) then merge with weights 8,1,1 is the 10-way mean
  task-arithmetic  task vectors add, so stage 1 at scale 1.0 then stage 2 at
                   scale S gives base + S * sum(all ten task vectors)
  ties             does NOT compose -- trimming keeps the top fraction of each
                   task vector and the sign election runs across the set, so
                   8-then-2 is not the same operation as 10 at once

So ties runs natively over the first eight domains, with a linear merge of the
same eight as its control, and science and sql held out as domains no merged
model in that pair ever saw.

Usage:
    uv run experiments/multidomain/merge_experiment.py

Logs:
    experiments/multidomain/logs/merge.log
"""

import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

SCRIPT_DIR = Path(__file__).parent
BASE_MODEL = SCRIPT_DIR / "base_model"
DB_DIR = SCRIPT_DIR / "reminis_dbs"
MERGED_DB_DIR = SCRIPT_DIR / "merged_dbs"
MERGED_MODEL_DIR = SCRIPT_DIR / "merged_models"
DATASETS_DIR = SCRIPT_DIR / "datasets"
PPL_RESULTS = SCRIPT_DIR / "perplexity_results.json"
RESULTS_FILE = SCRIPT_DIR / "merge_results.json"
LOG_FILE = SCRIPT_DIR / "logs" / "merge.log"

DOMAINS = ["code", "creative", "finance", "history", "legal",
           "math", "medical", "reasoning", "science", "sql"]

# The eight ties/linear8 see, and the two they never do.
EIGHT = DOMAINS[:8]
HELD_OUT = DOMAINS[8:]

MAX_TEST_SAMPLES = 30
MAX_SEQ_LEN = 512

# Files a model directory needs beside the weights for mlx_lm to load it.
TOKENIZER_FILES = ["config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "vocab.json", "merges.txt"]


def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def section(title):
    log(f"{'=' * 74}")
    log(f"  {title}")
    log(f"{'=' * 74}")
    log()


def db(name):
    return str(DB_DIR / f"{name}.db")


def reminis(*args):
    result = subprocess.run(["uv", "run", "reminis"] + [str(a) for a in args],
                            capture_output=True, text=True)
    if result.returncode != 0:
        log(f"    ERROR: {result.stderr.strip().splitlines()[-1][:200]}")
        return False
    return True


def merge(out_name, inputs, method="linear", base=None, weights=None,
          scale=None, density=None):
    """Run one `reminis merge`, skipping it if the output already exists."""
    out = MERGED_DB_DIR / f"{out_name}.db"
    if out.exists():
        log(f"  {out_name}: already merged, skipping")
        return True

    args = ["merge"] + inputs + ["--method", method, "-o", str(out)]
    if base:
        args += ["--base", base]
    if weights:
        args += ["--weights", ",".join(str(w) for w in weights)]
    if scale is not None:
        args += ["--scale", str(scale)]
    if density is not None:
        args += ["--density", str(density)]

    start = time.time()
    ok = reminis(*args, "-q")
    if ok:
        size = out.stat().st_size / 1e6
        log(f"  {out_name}: merged in {time.time() - start:.1f}s ({size:.0f} MB)")
    return ok


def export_model(name):
    """Turn a merged database into a directory mlx_lm can load."""
    out_dir = MERGED_MODEL_DIR / name
    if (out_dir / "model.safetensors").exists():
        log(f"  {name}: already exported, skipping")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    ok = reminis("export", str(MERGED_DB_DIR / f"{name}.db"),
                 "-o", str(out_dir / "model.safetensors"), "-q")
    if not ok:
        return False
    for f in TOKENIZER_FILES:
        src = BASE_MODEL / f
        if src.exists():
            shutil.copy(src, out_dir / f)
    log(f"  {name}: exported")
    return True


def compute_perplexity(model, tokenizer, texts):
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LEN]
        x, y = mx.array([tokens[:-1]]), mx.array([tokens[1:]])
        loss = nn.losses.cross_entropy(model(x), y, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += len(tokens) - 1
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def load_test_texts(domain, tokenizer):
    for fname in ("test.jsonl", "valid.jsonl"):
        path = DATASETS_DIR / domain / fname
        if path.exists():
            break
    else:
        return []

    texts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            formatted = tokenizer.apply_chat_template(
                json.loads(line).get("messages", []),
                tokenize=False, add_generation_prompt=False,
            )
            if len(formatted) > 20:
                texts.append(formatted)
            if len(texts) >= MAX_TEST_SAMPLES:
                break
    return texts


def fmt(ppl):
    if ppl == float("inf"):
        return "   inf"
    if ppl > 9999:
        return f"{ppl:6.0f}"
    return f"{ppl:6.1f}"


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")
    MERGED_DB_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    section("MERGE EXPERIMENT: ten fine-tunes into one model")

    eight_dbs = [db(d) for d in EIGHT]
    all_ok = True

    # ---- Stage 1: the eight-way merges -------------------------------------
    log("Stage 1 - eight-way merges")
    all_ok &= merge("linear8", eight_dbs, method="linear")
    all_ok &= merge("ties8", eight_dbs, method="ties",
                    base=db("base"), density=0.2, scale=1.0)
    # Intermediate for the ten-way task arithmetic: base + sum(tv_1..8).
    all_ok &= merge("_ta8_raw", eight_dbs, method="task-arithmetic",
                    base=db("base"), scale=1.0)
    log()

    # ---- Stage 2: fold in the last two -------------------------------------
    log("Stage 2 - folding in science and sql")
    held = [db(d) for d in HELD_OUT]

    # avg(8) weighted 8 against the two singles is exactly the ten-way mean.
    all_ok &= merge("linear10", [str(MERGED_DB_DIR / "linear8.db")] + held,
                    method="linear", weights=[8, 1, 1])

    # Task vectors add, so stage 1 at scale 1.0 then S here gives S * sum(ten).
    ta8 = str(MERGED_DB_DIR / "_ta8_raw.db")
    for tag, s in (("ta10_s01", 0.1), ("ta10_s03", 0.3), ("ta10_s10", 1.0)):
        all_ok &= merge(tag, [ta8] + held, method="task-arithmetic",
                        base=db("base"), scale=s)
    log()

    if not all_ok:
        log("Some merges failed; benchmarking whatever succeeded.")
        log()

    # ---- Export ------------------------------------------------------------
    section("EXPORT: merged databases to loadable models")
    benchmark_names = ["linear8", "ties8", "linear10",
                       "ta10_s01", "ta10_s03", "ta10_s10"]
    available = []
    for name in benchmark_names:
        if not (MERGED_DB_DIR / f"{name}.db").exists():
            log(f"  {name}: no database, skipping")
            continue
        if export_model(name):
            available.append(name)
    log()

    # ---- Benchmark ---------------------------------------------------------
    section("PERPLEXITY: each merged model on all ten domains")

    log("Loading test splits...")
    _, tok = load(str(BASE_MODEL))
    domain_texts = {d: load_test_texts(d, tok) for d in DOMAINS}
    del tok
    for d in DOMAINS:
        log(f"  {d}: {len(domain_texts[d])} samples")
    log()

    results = {}
    for i, name in enumerate(available):
        log(f"  [{i + 1}/{len(available)}] {name}")
        model, tokenizer = load(str(MERGED_MODEL_DIR / name))
        scores = {}
        start = time.time()
        for d in DOMAINS:
            texts = domain_texts[d]
            if not texts:
                scores[d] = float("inf")
                continue
            ppl = compute_perplexity(model, tokenizer, texts)
            scores[d] = round(ppl, 2)
            unseen = " (held out)" if name in ("linear8", "ties8") and d in HELD_OUT else ""
            log(f"      {d:12s}  ppl={fmt(ppl)}{unseen}")
        finite = [v for v in scores.values() if v != float("inf")]
        log(f"      -> avg {fmt(sum(finite) / len(finite))} in {time.time() - start:.0f}s")
        results[name] = scores
        del model
        log()

    # ---- Comparison --------------------------------------------------------
    section("RESULTS")

    prior = json.loads(PPL_RESULTS.read_text()) if PPL_RESULTS.exists() else {}
    base = prior.get("base", {})
    # The diagonal: each domain scored by the model trained on it.
    specialist = {d: prior.get(f"ft_{d}", {}).get(d) for d in DOMAINS}

    header = f"  {'Model':14s}" + "".join(f"{d[:7]:>9s}" for d in DOMAINS) + f"{'AVG':>9s}"
    log(header)
    log(f"  {'-' * (len(header) - 2)}")

    def row(label, scores):
        vals = [scores.get(d) for d in DOMAINS]
        line = f"  {label:14s}" + "".join(
            fmt(v).rjust(9) if v is not None else "        -" for v in vals)
        finite = [v for v in vals if v is not None and v != float("inf")]
        line += fmt(sum(finite) / len(finite)).rjust(9) if finite else "        -"
        log(line)

    if base:
        row("base", base)
    if all(v is not None for v in specialist.values()):
        row("specialists", specialist)
    for name in available:
        row(name, results[name])

    log()
    log("  'specialists' is the diagonal: each domain scored by the one model")
    log("  trained on it. It is ten models, not one, and is the bar a single")
    log("  merged model is trying to reach.")

    # Merge vs base and vs specialists, on average.
    if base and available:
        log()
        base_avg = sum(base[d] for d in DOMAINS) / len(DOMAINS)
        spec_finite = [v for v in specialist.values() if v]
        spec_avg = sum(spec_finite) / len(spec_finite) if spec_finite else None
        log(f"  base average:        {fmt(base_avg)}")
        if spec_avg:
            log(f"  specialist average:  {fmt(spec_avg)}  ({(spec_avg / base_avg - 1) * 100:+.1f}% vs base)")
        log()
        for name in available:
            vals = [results[name][d] for d in DOMAINS if results[name][d] != float("inf")]
            avg = sum(vals) / len(vals)
            log(f"  {name:14s} {fmt(avg)}  ({(avg / base_avg - 1) * 100:+.1f}% vs base)")

    RESULTS_FILE.write_text(json.dumps({
        "merged": results,
        "base": base,
        "specialists": specialist,
        "held_out_for_eight_way": HELD_OUT,
    }, indent=2))
    log()
    log(f"  Results saved to {RESULTS_FILE}")
    log(f"  Log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
