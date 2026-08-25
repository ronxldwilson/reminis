"""Can a merge beat the specialist on the specialist's own domain?

A plain average loses to every specialist because it dilutes that domain's task
vector by ten. But the ten fine-tunes share something real -- each one improved
every other domain -- and a specialist trained on 89 legal examples never saw
enough data to learn it.

So keep the domain's task vector at full strength and add only a fraction of
the rest:

    base + tv_d + alpha * sum(tv_others)

`ta10_s10.db` is already `base + sum(all ten tv)`, so this needs two inputs
rather than ten:

    base + (1-alpha) * tv_d + alpha * sum(all)
      = base + tv_d + alpha * sum(others)

which is a task-arithmetic merge of [domain, ta10_s10] weighted
[(1-alpha), alpha] at scale 1.0.

Usage:
    uv run experiments/multidomain/specialist_plus.py

Logs:
    experiments/multidomain/logs/specialist_plus.log
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
RESULTS_FILE = SCRIPT_DIR / "specialist_plus_results.json"
LOG_FILE = SCRIPT_DIR / "logs" / "specialist_plus.log"

DOMAINS = ["code", "creative", "finance", "history", "legal",
           "math", "medical", "reasoning", "science", "sql"]

# legal has the biggest gap to its specialist and the smallest training split;
# sql the second biggest; code the smallest, as a control where there is
# little room to improve.
TARGETS = ["legal", "sql", "code"]
ALPHAS = [0.05, 0.1, 0.2]

MAX_TEST_SAMPLES = 30
MAX_SEQ_LEN = 512
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
    log("=" * 74)
    log(f"  {title}")
    log("=" * 74)
    log()


def reminis(*args):
    r = subprocess.run(["uv", "run", "reminis"] + [str(a) for a in args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        tail = r.stderr.strip().splitlines()
        log(f"    ERROR: {tail[-1][:200] if tail else 'unknown'}")
        return False
    return True


def build(domain, alpha):
    """base + tv_domain + alpha * sum(task vectors of the other nine)."""
    name = f"sp_{domain}_a{str(alpha).replace('.', '')}"
    out = MERGED_DB_DIR / f"{name}.db"
    if not out.exists():
        ok = reminis(
            "merge",
            str(DB_DIR / f"{domain}.db"),
            str(MERGED_DB_DIR / "ta10_s10.db"),
            "--method", "task-arithmetic",
            "--base", str(DB_DIR / "base.db"),
            "--weights", f"{1 - alpha},{alpha}",
            "--scale", "1.0",
            "-o", str(out), "-q",
        )
        if not ok:
            return None
    model_dir = MERGED_MODEL_DIR / name
    if not (model_dir / "model.safetensors").exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        if not reminis("export", str(out), "-o", str(model_dir / "model.safetensors"), "-q"):
            return None
        for f in TOKENIZER_FILES:
            src = BASE_MODEL / f
            if src.exists():
                shutil.copy(src, model_dir / f)
    return name


def perplexity(model, tokenizer, texts):
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        tokens = tokenizer.encode(text)[:MAX_SEQ_LEN]
        if len(tokens) < 2:
            continue
        x, y = mx.array([tokens[:-1]]), mx.array([tokens[1:]])
        loss = nn.losses.cross_entropy(model(x), y, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += len(tokens) - 1
    return math.exp(total_loss / total_tokens) if total_tokens else float("inf")


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
                tokenize=False, add_generation_prompt=False)
            if len(formatted) > 20:
                texts.append(formatted)
            if len(texts) >= MAX_TEST_SAMPLES:
                break
    return texts


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")

    section("SPECIALIST + a fraction of everything else")
    log("  base + tv_domain + alpha * sum(other nine task vectors)")
    log(f"  domains: {', '.join(TARGETS)}   alphas: {ALPHAS}")
    log()

    if not (MERGED_DB_DIR / "ta10_s10.db").exists():
        log("  ta10_s10.db missing -- run merge_experiment.py first.")
        return

    prior = json.loads(PPL_RESULTS.read_text())
    merge_prior = json.loads((SCRIPT_DIR / "merge_results.json").read_text())
    plain_merge = merge_prior["merged"]["linear10"]

    log("Building merges...")
    built = {}
    for d in TARGETS:
        for a in ALPHAS:
            start = time.time()
            name = build(d, a)
            if name:
                built[(d, a)] = name
                log(f"  {d:8s} alpha={a:<5} -> {name}  ({time.time() - start:.1f}s)")
    log()

    log("Loading test splits...")
    _, tok = load(str(BASE_MODEL))
    texts = {d: load_test_texts(d, tok) for d in DOMAINS}
    del tok
    log()

    section("MEASURING")
    results = {}
    for (d, a), name in built.items():
        model, tokenizer = load(str(MERGED_MODEL_DIR / name))
        scores = {dom: round(perplexity(model, tokenizer, texts[dom]), 2)
                  for dom in DOMAINS}
        results[name] = {"domain": d, "alpha": a, "scores": scores}
        own = scores[d]
        spec = prior[f"ft_{d}"][d]
        avg = sum(scores.values()) / len(scores)
        verdict = "BEATS specialist" if own < spec else "loses"
        log(f"  {d:8s} alpha={a:<5} own={own:6.2f} (specialist {spec:5.2f}) "
            f"avg={avg:5.2f}  {verdict}")
        del model
    log()

    section("RESULTS")
    log("  For each domain: perplexity on its OWN test split.")
    log()
    header = (f"  {'domain':10s}{'base':>8s}{'plain':>8s}{'special':>9s}"
              + "".join(f"{'a=' + str(a):>9s}" for a in ALPHAS))
    log(header)
    log("  " + "-" * (len(header) - 2))
    for d in TARGETS:
        spec = prior[f"ft_{d}"][d]
        row = (f"  {d:10s}{prior['base'][d]:8.2f}{plain_merge[d]:8.2f}"
               f"{spec:9.2f}")
        for a in ALPHAS:
            name = built.get((d, a))
            row += f"{results[name]['scores'][d]:9.2f}" if name else "        -"
        log(row)
    log()
    log("  'plain' is the ten-way linear merge; 'special' the model trained")
    log("  on that domain alone. a=* keeps that domain's task vector whole and")
    log("  adds alpha of the other nine.")
    log()

    log("  And the same models as generalists (mean over all ten domains):")
    log()
    header2 = f"  {'domain':10s}{'special':>9s}" + "".join(f"{'a=' + str(a):>9s}" for a in ALPHAS)
    log(header2)
    log("  " + "-" * (len(header2) - 2))
    for d in TARGETS:
        spec_avg = sum(prior[f"ft_{d}"][x] for x in DOMAINS) / len(DOMAINS)
        row = f"  {d:10s}{spec_avg:9.2f}"
        for a in ALPHAS:
            name = built.get((d, a))
            if name:
                s = results[name]["scores"]
                row += f"{sum(s.values()) / len(s):9.2f}"
            else:
                row += "        -"
        log(row)

    RESULTS_FILE.write_text(json.dumps({
        "results": results,
        "specialists": {d: prior[f"ft_{d}"][d] for d in TARGETS},
        "base": {d: prior["base"][d] for d in TARGETS},
        "plain_merge": {d: plain_merge[d] for d in TARGETS},
    }, indent=2))
    log()
    log(f"  Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
