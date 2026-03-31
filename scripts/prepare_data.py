"""
Data preparation script: formats raw ObliQA dataset into MCQ-4 and MCQ-20.
Handles train/test splitting to prevent data leakage.

Usage:
    python scripts/prepare_data.py \
        --input data/raw/obliqa.json \
        --output data/processed/ \
        --mcq_sizes 4 20 \
        --test_split 0.2 \
        --seed 42
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict


def load_dataset(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def format_mcq(sample: Dict, num_options: int, all_samples: List[Dict]) -> Dict:
    """
    Format a sample as MCQ with `num_options` choices.
    Extra distractors are sampled from other answer texts in the dataset.
    """
    correct = sample["answer"]
    options = [correct]

    # Sample distractors from other answers in the dataset
    distractors = [
        s["answer"] for s in all_samples
        if s["answer"] != correct and s["answer"] not in options
    ]
    random.shuffle(distractors)
    options += distractors[: num_options - 1]

    if len(options) < num_options:
        # Pad if not enough distractors
        options += [f"None of the above (option {i})" for i in range(num_options - len(options))]

    random.shuffle(options)
    correct_label = chr(65 + options.index(correct))  # A, B, C...

    return {
        "id": sample.get("id", ""),
        "question": sample["question"],
        "options": options,
        "answer": correct_label,
        "answer_text": correct,
        "mcq_size": num_options,
        "source": sample.get("source", ""),
    }


def split_dataset(samples, test_size=0.2, seed=42):
    random.seed(seed)
    shuffled = samples[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - test_size))
    return shuffled[:split], shuffled[split:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mcq_sizes", nargs="+", type=int, default=[4, 20])
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    raw = load_dataset(args.input)
    print(f"Loaded {len(raw)} raw samples")

    train_raw, test_raw = split_dataset(raw, args.test_split, args.seed)
    print(f"Train: {len(train_raw)}, Test: {len(test_raw)}")

    for size in args.mcq_sizes:
        for split_name, split_data in [("train", train_raw), ("test", test_raw)]:
            formatted = [format_mcq(s, size, raw) for s in split_data]
            out_path = Path(args.output) / f"mcq{size}_{split_name}.json"
            with open(out_path, "w") as f:
                json.dump(formatted, f, indent=2)
            print(f"Saved MCQ-{size} {split_name}: {len(formatted)} samples → {out_path}")


if __name__ == "__main__":
    main()
