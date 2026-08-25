"""Benchmark base + 10 fine-tuned models, storage analysis via reminis.

Pipeline:
  1. mlx_lm fuse     — merge each LoRA adapter into a standalone model
  2. reminis convert  — all 11 models → reminis DBs
  3. reminis diff     — measure what each fine-tune changed
  4. mlx_lm generate  — benchmark text generation on all domains
  5. Storage analysis — separate files vs reminis delta packs

Usage:
    uv run experiments/multidomain/benchmark.py
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASE_MODEL = SCRIPT_DIR / "base_model"
ADAPTERS_DIR = SCRIPT_DIR / "adapters"
FUSED_DIR = SCRIPT_DIR / "fused_models"
DB_DIR = SCRIPT_DIR / "reminis_dbs"
RESULTS_FILE = SCRIPT_DIR / "benchmark_results.json"

DOMAINS = ["code", "creative", "finance", "history", "legal",
           "math", "medical", "reasoning", "science", "sql"]

DOMAIN_PROMPTS = {
    "medical": "What are the common symptoms and treatment options for Type 2 diabetes?",
    "legal": "Does a contract clause stating payment within 30 days create a breach if payment arrives on day 35?",
    "code": "Write a Python function that implements binary search on a sorted list.",
    "math": "A store sells apples for $2 each. If Sarah buys 5 apples and pays with a $20 bill, how much change does she get? Show your work step by step.",
    "finance": "Explain the difference between a stock and a bond, and when an investor might prefer one over the other.",
    "science": "What is the difference between covalent and ionic bonds? Give examples of each.",
    "sql": "Given a table employees(id, name, salary, department), write SQL to find the average salary per department ordered by highest average first.",
    "creative": "Write a short story about a child who discovers a hidden door in their grandmother's house.",
    "history": "What role did the Silk Road play in spreading religions across Asia?",
    "reasoning": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
}

DOMAIN_KEYWORDS = {
    "medical": ["diabetes", "insulin", "glucose", "blood sugar", "treatment", "symptom"],
    "legal": ["breach", "contract", "obligation", "damages", "performance", "pay"],
    "code": ["def ", "return", "while", "if ", "mid", "search"],
    "math": ["change", "=", "5", "10", "apples", "20"],
    "finance": ["stock", "bond", "invest", "risk", "return", "dividend"],
    "science": ["electron", "bond", "ionic", "covalent", "atom", "share"],
    "sql": ["select", "from", "group by", "order by", "avg", "salary"],
    "creative": ["door", "house", "grandmother", "child", "discover", "found"],
    "history": ["silk road", "buddhism", "islam", "christianity", "religion", "trade"],
    "reasoning": ["conclude", "roses", "flowers", "some", "all", "logic"],
}

R = "\033[0m"
B = "\033[1m"
G = "\033[32m"
Y = "\033[33m"
C = "\033[36m"
D = "\033[2m"
RED = "\033[31m"


def run_cmd(args, label=None):
    if label:
        print(f"  {D}$ {' '.join(str(a) for a in args[:8])}...{R}")
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if result.returncode != 0 and result.stderr:
        print(f"  {RED}ERROR: {result.stderr[:300]}{R}")
    return result


def score_response(response, domain):
    low = response.lower()
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    if not keywords:
        return 0.0
    return sum(1 for kw in keywords if kw.lower() in low) / len(keywords)


def get_size(path):
    p = Path(path)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size if p.exists() else 0


def fmt(b):
    if b >= 1e9: return f"{b/1e9:.2f} GB"
    if b >= 1e6: return f"{b/1e6:.2f} MB"
    return f"{b/1e3:.2f} KB"


def section(title):
    print(f"\n{B}{'═' * 62}{R}")
    print(f"{B}  {title}{R}")
    print(f"{B}{'═' * 62}{R}\n")


def main():
    FUSED_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    storage = {}

    # ─── Step 1: Fuse LoRA adapters into standalone models ───
    section("STEP 1: Fuse LoRA adapters → standalone models")

    python = str(Path(sys.executable))
    for domain in DOMAINS:
        fused = FUSED_DIR / domain
        adapter = ADAPTERS_DIR / domain
        if fused.exists() and any(fused.glob("*.safetensors")):
            print(f"  {G}✓ {domain}: already fused{R}")
            continue
        if not (adapter / "adapters.safetensors").exists():
            print(f"  {RED}✗ {domain}: no adapter{R}")
            continue
        print(f"  {Y}▶ {domain}: fusing...{R}", end="", flush=True)
        run_cmd([python, "-m", "mlx_lm", "fuse",
                 "--model", str(BASE_MODEL),
                 "--adapter-path", str(adapter),
                 "--save-path", str(fused)])
        if any(fused.glob("*.safetensors")):
            print(f"\r  {G}✓ {domain}: fused ({fmt(get_size(fused))}){R}")
        else:
            print(f"\r  {RED}✗ {domain}: fuse failed{R}")

    # ─── Step 2: Convert all models to reminis DBs ───
    section("STEP 2: reminis convert → SQLite databases")

    base_db = DB_DIR / "base.db"
    if not base_db.exists():
        print(f"  Converting base model...")
        run_cmd(["uv", "run", "reminis", "convert", str(BASE_MODEL), "-o", str(base_db)])
    print(f"  {G}✓ base.db: {fmt(get_size(base_db))}{R}")

    for domain in DOMAINS:
        domain_db = DB_DIR / f"{domain}.db"
        fused = FUSED_DIR / domain
        if domain_db.exists():
            print(f"  {G}✓ {domain}.db: {fmt(get_size(domain_db))}{R}")
            continue
        if not fused.exists():
            continue
        print(f"  {Y}▶ {domain}: converting...{R}", end="", flush=True)
        run_cmd(["uv", "run", "reminis", "convert", str(fused), "-o", str(domain_db)])
        print(f"\r  {G}✓ {domain}.db: {fmt(get_size(domain_db))}{R}")

    # ─── Step 3: reminis diff — what changed per fine-tune ───
    section("STEP 3: reminis diff — what each fine-tune changed")

    diff_results = {}
    for domain in DOMAINS:
        domain_db = DB_DIR / f"{domain}.db"
        if not domain_db.exists():
            continue
        result = run_cmd(["uv", "run", "reminis", "diff", str(base_db), str(domain_db)])
        output = result.stdout.strip() if result.returncode == 0 else "error"
        diff_results[domain] = output

        lines = output.split("\n")
        changed = [l for l in lines if "Changed" in l]
        summary = changed[0].strip() if changed else lines[1].strip() if len(lines) > 1 else ""
        print(f"  {B}{domain:12s}{R} {summary}")

        # Also generate delta pack
        delta = DB_DIR / f"{domain}_delta.bin"
        if not delta.exists():
            run_cmd(["uv", "run", "reminis", "diff", str(base_db), str(domain_db),
                     "-o", str(delta)], label=f"delta {domain}")

    # ─── Step 4: Benchmark with mlx_lm ───
    section("STEP 4: Benchmark — generate on each model × each domain")

    from mlx_lm import load, generate

    def bench_model(model_path, name):
        print(f"\n  {C}Loading: {name}{R}")
        model, tokenizer = load(str(model_path))
        scores = {}

        for domain, prompt in DOMAIN_PROMPTS.items():
            messages = [{"role": "user", "content": prompt}]
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            start = time.time()
            response = generate(model, tokenizer, prompt=chat_prompt,
                                max_tokens=256, verbose=False)
            elapsed = time.time() - start
            score = score_response(response, domain)
            scores[domain] = {
                "score": round(score, 3),
                "time": round(elapsed, 2),
                "response_length": len(response),
                "response_preview": response[:300],
            }
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"    {domain:12s} [{bar}] {score:.2f}  ({elapsed:.1f}s, {len(response)} chars)")

        del model
        return scores

    all_results["base"] = bench_model(BASE_MODEL, "Base (Qwen2.5-0.5B-Instruct)")

    for domain in DOMAINS:
        fused = FUSED_DIR / domain
        if fused.exists() and any(fused.glob("*.safetensors")):
            all_results[f"ft_{domain}"] = bench_model(fused, f"Fine-tuned: {domain}")

    # ─── Step 5: Storage analysis ───
    section("STEP 5: Storage analysis")

    base_sf_size = get_size(BASE_MODEL)
    base_db_size = get_size(base_db)

    print(f"  {B}Per-model sizes:{R}")
    print(f"    {'Model':14s} {'Safetensors':>12s} {'Reminis DB':>12s} {'Delta':>12s} {'Delta %':>8s}")
    print(f"    {'─' * 60}")
    print(f"    {'base':14s} {fmt(base_sf_size):>12s} {fmt(base_db_size):>12s} {'—':>12s} {'—':>8s}")

    total_sf = base_sf_size
    total_db = base_db_size
    total_delta = base_db_size  # base DB is always needed

    for domain in DOMAINS:
        fused = FUSED_DIR / domain
        domain_db = DB_DIR / f"{domain}.db"
        delta = DB_DIR / f"{domain}_delta.bin"

        sf_size = get_size(fused)
        db_size = get_size(domain_db)
        d_size = get_size(delta)

        total_sf += sf_size
        total_db += db_size
        total_delta += d_size

        pct = f"{d_size / sf_size * 100:.1f}%" if sf_size > 0 and d_size > 0 else "—"
        print(f"    {domain:14s} {fmt(sf_size):>12s} {fmt(db_size):>12s} {fmt(d_size):>12s} {pct:>8s}")

    print(f"    {'─' * 60}")
    print(f"    {'TOTAL':14s} {fmt(total_sf):>12s} {fmt(total_db):>12s} {fmt(total_delta):>12s}")

    print(f"\n  {B}Storage comparison (11 models: 1 base + 10 fine-tunes):{R}")
    print(f"    Approach A — 11 separate safetensors:  {fmt(total_sf)}")
    print(f"    Approach B — 11 separate reminis DBs:  {fmt(total_db)}")
    print(f"    Approach C — 1 base DB + 10 deltas:    {fmt(total_delta)}")
    if total_sf > 0:
        sav_b = (1 - total_db / total_sf) * 100
        sav_c = (1 - total_delta / total_sf) * 100
        print(f"\n    {G}Savings B vs A: {sav_b:.1f}%{R}")
        print(f"    {G}Savings C vs A: {sav_c:.1f}%{R}")
        if total_db > 0:
            sav_c_vs_b = (1 - total_delta / total_db) * 100
            print(f"    {G}Savings C vs B: {sav_c_vs_b:.1f}%{R}")

    storage = {
        "base_safetensors": base_sf_size,
        "base_db": base_db_size,
        "total_safetensors": total_sf,
        "total_dbs": total_db,
        "total_delta": total_delta,
        "per_domain": {d: {"delta": get_size(DB_DIR / f"{d}_delta.bin")} for d in DOMAINS},
    }

    # ─── Score matrix ───
    section("RESULTS: Score matrix (rows=models, cols=domains)")

    domains = list(DOMAIN_PROMPTS.keys())
    header = f"{'Model':16s}" + "".join(f"{d[:6]:>8s}" for d in domains) + f"{'AVG':>8s}"
    print(f"  {B}{header}{R}")
    print(f"  {'─' * len(header)}")

    for model_name in ["base"] + [f"ft_{d}" for d in DOMAINS]:
        if model_name not in all_results:
            continue
        scores = all_results[model_name]
        row = f"  {model_name:16s}"
        vals = []
        for d in domains:
            s = scores.get(d, {}).get("score", 0)
            vals.append(s)
            is_own = model_name == f"ft_{d}"
            c = G if is_own else ""
            e = R if is_own else ""
            row += f"{c}{s:8.2f}{e}"
        avg = sum(vals) / len(vals) if vals else 0
        row += f"{avg:8.2f}"
        print(row)

    # ─── Save ───
    output = {
        "benchmarks": all_results,
        "storage": storage,
        "diffs": {d: diff_results.get(d, "")[:500] for d in DOMAINS},
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
