"""Download and format 10 diverse domain datasets for mlx_lm LoRA fine-tuning."""

import json
import os
from pathlib import Path
from datasets import load_dataset

BASE = Path(__file__).parent / "datasets"

DOMAINS = {
    "medical": {
        "hf": "medalpaca/medical_meadow_medical_flashcards",
        "format": lambda ex: {
            "messages": [
                {"role": "user", "content": ex["input"]},
                {"role": "assistant", "content": ex["output"]},
            ]
        },
    },
    "legal": {
        "hf": "nguha/legalbench",
        "subset": "contract_nli_explicit_identification",
        "format": lambda ex: {
            "messages": [
                {"role": "user", "content": f"Legal analysis: {ex['text']}"},
                {"role": "assistant", "content": ex["answer"]},
            ]
        },
    },
    "code": {
        "hf": "iamtarun/python_code_instructions_18k_alpaca",
        "format": lambda ex: {
            "messages": [
                {"role": "user", "content": ex["instruction"] + ("\n" + ex["input"] if ex.get("input") else "")},
                {"role": "assistant", "content": ex["output"]},
            ]
        },
    },
    "math": {
        "hf": "openai/gsm8k",
        "subset": "main",
        "format": lambda ex: {
            "messages": [
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]},
            ]
        },
    },
    "finance": {
        "hf": "gbharti/finance-alpaca",
        "format": lambda ex: {
            "messages": [
                {"role": "user", "content": ex["instruction"] + ("\n" + ex["input"] if ex.get("input") else "")},
                {"role": "assistant", "content": ex["output"]},
            ]
        },
    },
    "science": {
        "hf": "derek-thomas/ScienceQA",
        "format": lambda ex: {
            "messages": [
                {"role": "user", "content": ex.get("question", "") + ("\nChoices: " + str(ex.get("choices", "")) if ex.get("choices") else "")},
                {"role": "assistant", "content": ex.get("solution", "") or ex.get("lecture", "")},
            ]
        },
    },
    "sql": {
        "hf": "b-mc2/sql-create-context",
        "format": lambda ex: {
            "messages": [
                {"role": "user", "content": "Context: " + ex.get("context", "") + "\nQuestion: " + ex.get("question", "")},
                {"role": "assistant", "content": ex.get("answer", "")},
            ]
        },
    },
    "creative": {
        "hf": "roneneldan/TinyStories",
        "format": lambda ex: {
            "messages": [
                {"role": "user", "content": "Write a short story."},
                {"role": "assistant", "content": ex.get("text", "")},
            ]
        },
    },
    "history": {
        "hf": "cais/mmlu",
        "subset": "world_religions",
        "format": lambda ex: {
            "messages": [
                {"role": "user", "content": ex["question"] + "\nA) " + ex["choices"][0] + "\nB) " + ex["choices"][1] + "\nC) " + ex["choices"][2] + "\nD) " + ex["choices"][3]},
                {"role": "assistant", "content": "The answer is " + ["A", "B", "C", "D"][ex["answer"]] + ") " + ex["choices"][ex["answer"]] + "."},
            ]
        },
    },
    "reasoning": {
        "hf": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "format": lambda ex: {
            "messages": [
                {"role": "user", "content": ex["question"] + "\n" + "\n".join(f"{l}) {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"]))},
                {"role": "assistant", "content": "The answer is " + ex["answerKey"] + ". " + ex["question"]},
            ]
        },
    },
}

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


def main():
    for domain, cfg in DOMAINS.items():
        print(f"\n{'='*60}")
        print(f"Preparing: {domain}")
        print(f"{'='*60}")

        out_dir = BASE / domain
        if (out_dir / "train.jsonl").exists():
            print(f"  Already exists, skipping")
            continue

        try:
            kwargs = {"split": "train"}
            if "subset" in cfg:
                ds = load_dataset(cfg["hf"], cfg["subset"], **kwargs, trust_remote_code=False)
            else:
                ds = load_dataset(cfg["hf"], **kwargs, trust_remote_code=False)

            total_needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
            if len(ds) > total_needed:
                ds = ds.shuffle(seed=42).select(range(total_needed))

            formatted = [cfg["format"](ex) for ex in ds]
            formatted = [ex for ex in formatted if ex["messages"][1]["content"] and len(ex["messages"][1]["content"].strip()) > 10]

            if len(formatted) < 100:
                print(f"  WARNING: only {len(formatted)} valid examples, skipping")
                continue

            train = formatted[:TRAIN_SIZE]
            val = formatted[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
            test = formatted[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

            save_split(train, out_dir / "train.jsonl")
            save_split(val, out_dir / "valid.jsonl")
            save_split(test, out_dir / "test.jsonl")

            print(f"  Saved: {len(train)} train, {len(val)} valid, {len(test)} test")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
