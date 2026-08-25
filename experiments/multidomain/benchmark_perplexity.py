"""Cross-domain perplexity benchmark: each model × each domain's test set.

Lower perplexity = model is better at predicting that domain's text.
A fine-tuned model should have much lower perplexity on its own domain
than the base model does.

Usage:
    uv run experiments/multidomain/benchmark_perplexity.py

Logs:
    experiments/multidomain/logs/perplexity.log
"""

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

# Force unbuffered stdout so backgrounded runs show progress
sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

SCRIPT_DIR = Path(__file__).parent
BASE_MODEL = SCRIPT_DIR / "base_model"
FUSED_DIR = SCRIPT_DIR / "fused_models"
DATASETS_DIR = SCRIPT_DIR / "datasets"
RESULTS_FILE = SCRIPT_DIR / "perplexity_results.json"
LOG_FILE = SCRIPT_DIR / "logs" / "perplexity.log"

DOMAINS = ["code", "creative", "finance", "history", "legal",
           "math", "medical", "reasoning", "science", "sql"]

MAX_TEST_SAMPLES = 30
MAX_SEQ_LEN = 512


def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def section(title):
    log(f"{'═' * 70}")
    log(f"  {title}")
    log(f"{'═' * 70}")
    log()


def compute_perplexity(model, tokenizer, texts, domain_name=""):
    total_loss = 0.0
    total_tokens = 0

    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN]

        x = mx.array([tokens[:-1]])
        y = mx.array([tokens[1:]])

        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += len(tokens) - 1

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def load_test_texts(domain, tokenizer):
    test_file = DATASETS_DIR / domain / "test.jsonl"
    if not test_file.exists():
        test_file = DATASETS_DIR / domain / "valid.jsonl"
    if not test_file.exists():
        return []

    texts = []
    with open(test_file) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            messages = ex.get("messages", [])
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if len(formatted) > 20:
                texts.append(formatted)
            if len(texts) >= MAX_TEST_SAMPLES:
                break
    return texts


def fmt_ppl(ppl):
    if ppl == float("inf"):
        return "   inf"
    if ppl > 9999:
        return f"{ppl:6.0f}"
    return f"{ppl:6.1f}"


def main():
    # Clear previous log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        f.write("")

    section("CROSS-DOMAIN PERPLEXITY BENCHMARK")
    log(f"  Models: base + 10 fine-tuned")
    log(f"  Test sets: {MAX_TEST_SAMPLES} samples/domain, max {MAX_SEQ_LEN} tokens each")
    log(f"  Lower perplexity = better fit for that domain")
    log()

    all_results = {}
    models_to_test = [("base", BASE_MODEL)]
    for domain in DOMAINS:
        fused = FUSED_DIR / domain
        if fused.exists() and any(fused.glob("*.safetensors")):
            models_to_test.append((f"ft_{domain}", fused))

    log(f"  Found {len(models_to_test)} models to test")
    log()

    # Preload all test texts using base tokenizer
    log("  Loading tokenizer for test data...")
    _, base_tokenizer = load(str(BASE_MODEL))
    domain_texts = {}
    for domain in DOMAINS:
        texts = load_test_texts(domain, base_tokenizer)
        domain_texts[domain] = texts
        log(f"    {domain}: {len(texts)} test samples")
    del base_tokenizer
    log()

    total_models = len(models_to_test)
    start_all = time.time()

    for mi, (model_name, model_path) in enumerate(models_to_test):
        log(f"  ──────────────────────────────────────────")
        log(f"  [{mi+1}/{total_models}] Loading: {model_name}")
        log(f"  ──────────────────────────────────────────")
        start_load = time.time()
        model, tokenizer = load(str(model_path))
        load_time = time.time() - start_load
        log(f"    loaded in {load_time:.1f}s")

        model_scores = {}
        model_start = time.time()

        for di, domain in enumerate(DOMAINS):
            texts = domain_texts[domain]
            if not texts:
                model_scores[domain] = float("inf")
                log(f"    {domain:12s}  ppl=  inf   (no test data)")
                continue

            start = time.time()
            ppl = compute_perplexity(model, tokenizer, texts, domain)
            elapsed = time.time() - start

            model_scores[domain] = round(ppl, 2)

            is_own = model_name == f"ft_{domain}"
            marker = " ★" if is_own else "  "
            log(f"  {marker} {domain:12s}  ppl={fmt_ppl(ppl)}  ({elapsed:.1f}s, {len(texts)} samples)")

        model_elapsed = time.time() - model_start
        avg_ppl = sum(v for v in model_scores.values() if v != float("inf")) / max(1, sum(1 for v in model_scores.values() if v != float("inf")))
        log(f"    model done in {model_elapsed:.1f}s — avg perplexity: {fmt_ppl(avg_ppl)}")

        all_results[model_name] = model_scores
        del model
        log()

    total_elapsed = time.time() - start_all
    log(f"  All models evaluated in {total_elapsed:.0f}s")

    # ─── Results matrix ───
    section("PERPLEXITY MATRIX (lower = better)")

    header = f"  {'Model':16s}" + "".join(f"{d[:7]:>9s}" for d in DOMAINS) + f"{'AVG':>9s}"
    log(header)
    log(f"  {'─' * (len(header) - 2)}")

    for model_name in ["base"] + [f"ft_{d}" for d in DOMAINS]:
        if model_name not in all_results:
            continue
        scores = all_results[model_name]
        row = f"  {model_name:16s}"
        vals = []
        for d in DOMAINS:
            ppl = scores.get(d, float("inf"))
            row += f"{fmt_ppl(ppl):>9s}"
            if ppl != float("inf"):
                vals.append(ppl)
        avg = sum(vals) / len(vals) if vals else float("inf")
        row += f"{fmt_ppl(avg):>9s}"
        log(row)

    # ─── Improvement over base ───
    section("PERPLEXITY CHANGE vs BASE (negative = improved)")

    if "base" in all_results:
        base_scores = all_results["base"]
        header2 = f"  {'Model':16s}" + "".join(f"{d[:7]:>9s}" for d in DOMAINS) + f"{'AVG':>9s}"
        log(header2)
        log(f"  {'─' * (len(header2) - 2)}")

        for model_name in [f"ft_{d}" for d in DOMAINS]:
            if model_name not in all_results:
                continue
            scores = all_results[model_name]
            row = f"  {model_name:16s}"
            deltas = []
            for d in DOMAINS:
                base_ppl = base_scores.get(d, float("inf"))
                ft_ppl = scores.get(d, float("inf"))
                if base_ppl == float("inf") or ft_ppl == float("inf"):
                    row += f"{'—':>9s}"
                else:
                    delta = ((ft_ppl - base_ppl) / base_ppl) * 100
                    deltas.append(delta)
                    row += f"{delta:>+8.1f}%"
            avg_delta = sum(deltas) / len(deltas) if deltas else 0
            row += f"{avg_delta:>+8.1f}%"
            log(row)

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    log()
    log(f"  Results saved to {RESULTS_FILE}")
    log(f"  Log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
