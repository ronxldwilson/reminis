"""Fix the 4 datasets that failed: legal, science, history, reasoning."""

import json
from pathlib import Path
from datasets import load_dataset

BASE = Path(__file__).parent / "datasets"
TRAIN_SIZE = 500
VAL_SIZE = 50
TEST_SIZE = 50


def save_split(examples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            content = ex["messages"][1]["content"]
            if content and len(content.strip()) > 10:
                f.write(json.dumps(ex) + "\n")


def prepare(domain, ds, format_fn):
    out_dir = BASE / domain
    if (out_dir / "train.jsonl").exists():
        print(f"  {domain}: already exists, skipping")
        return

    total_needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if len(ds) > total_needed:
        ds = ds.shuffle(seed=42).select(range(total_needed))

    formatted = [format_fn(ex) for ex in ds]
    formatted = [ex for ex in formatted if ex["messages"][1]["content"] and len(ex["messages"][1]["content"].strip()) > 10]

    if len(formatted) < 100:
        print(f"  {domain}: WARNING only {len(formatted)} valid examples")
        return

    train = formatted[:TRAIN_SIZE]
    val = formatted[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test = formatted[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    save_split(train, out_dir / "train.jsonl")
    save_split(val, out_dir / "valid.jsonl")
    save_split(test, out_dir / "test.jsonl")
    print(f"  {domain}: {len(train)} train, {len(val)} valid, {len(test)} test")


def main():
    print("=== Legal (contract_nli from legalbench, test split) ===")
    try:
        ds = load_dataset("nguha/legalbench", "contract_nli_explicit_identification", split="test")
        prepare("legal", ds, lambda ex: {
            "messages": [
                {"role": "user", "content": f"Does this contract clause explicitly identify the contracting parties? Clause: {ex['text']}"},
                {"role": "assistant", "content": f"Based on the contract clause analysis, the answer is: {ex['answer']}. The clause {'explicitly identifies' if ex['answer'] == 'Yes' else 'does not explicitly identify'} the contracting parties."},
            ]
        })
    except Exception as e:
        print(f"  Error: {e}")
        print("  Trying alternative: pile-of-law subset")
        ds = load_dataset("open-phi/textbooks", split="train")
        prepare("legal", ds, lambda ex: {
            "messages": [
                {"role": "user", "content": ex.get("textbook", "")[:300]},
                {"role": "assistant", "content": ex.get("textbook", "")[300:800]},
            ]
        })

    print("\n=== Science (sciq) ===")
    try:
        ds = load_dataset("allenai/sciq", split="train")
        prepare("science", ds, lambda ex: {
            "messages": [
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": f"The correct answer is: {ex['correct_answer']}. {ex['support']}"},
            ]
        })
    except Exception as e:
        print(f"  Error: {e}")

    print("\n=== History (MMLU, test split) ===")
    try:
        ds = load_dataset("cais/mmlu", "prehistory", split="test")
        prepare("history", ds, lambda ex: {
            "messages": [
                {"role": "user", "content": ex["question"] + "\nA) " + ex["choices"][0] + "\nB) " + ex["choices"][1] + "\nC) " + ex["choices"][2] + "\nD) " + ex["choices"][3]},
                {"role": "assistant", "content": "The answer is " + ["A", "B", "C", "D"][ex["answer"]] + ") " + ex["choices"][ex["answer"]] + ". " + ex["question"]},
            ]
        })
    except Exception as e:
        print(f"  Error: {e}")

    print("\n=== Reasoning (ARC, test split) ===")
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        prepare("reasoning", ds, lambda ex: {
            "messages": [
                {"role": "user", "content": ex["question"] + "\n" + "\n".join(f"{l}) {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"]))},
                {"role": "assistant", "content": f"The answer is {ex['answerKey']}. Let me explain: {ex['question']}"},
            ]
        })
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
