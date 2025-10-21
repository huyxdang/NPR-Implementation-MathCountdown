import os
import random
import json
from collections import defaultdict
from datasets import load_dataset
from math_verify import extract_answer  # Use math-verify instead

# --- CONFIG ---
DATASET_NAME = "huyxdang/MATH-dataset"
OUTPUT_DIR = "./data/math_json"
SUBSET_RATIO = 0.10
LABEL_COL = "type"
SEED = 42

# --- Ensure output dir ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load dataset ---
dataset = load_dataset(DATASET_NAME, trust_remote_code=True)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
print(f"Loaded {len(train_dataset)} train, {len(test_dataset)} test examples.")

# --- Stratified sampling ---
def stratified_sample(dataset, label_col, subset_ratio, seed=42):
    grouped = defaultdict(list)
    for i, ex in enumerate(dataset):
        grouped[ex[label_col]].append(i)
    
    sampled_indices = []
    random.seed(seed)
    for label, indices in grouped.items():
        n = int(len(indices) * subset_ratio)
        sampled_indices += random.sample(indices, max(1, n))
    
    return dataset.select(sampled_indices)

if SUBSET_RATIO < 1.0:
    train_dataset = stratified_sample(train_dataset, LABEL_COL, SUBSET_RATIO, SEED)
    test_dataset = stratified_sample(test_dataset, LABEL_COL, SUBSET_RATIO, SEED)
    print(f"⚖️ Stratified sampling done: {len(train_dataset)} train, {len(test_dataset)} test retained.")
else:
    print("⚠️ Using full dataset (no sampling).")

# --- Transform ---
def make_map_fn(split, data_source):
    def process_fn(example, idx):
        question = example["problem"]
        solution = example["solution"]
        answer = extract_answer(solution)  # Use math-verify's extract_answer
        
        ground_truth = {
            "question": question,
            "solution": solution,
            "target": answer,
        }
        
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                "split": split,
                "index": idx
            }
        }
    return process_fn

train_dataset = train_dataset.map(make_map_fn("train", DATASET_NAME), with_indices=True)
test_dataset = test_dataset.map(make_map_fn("test", DATASET_NAME), with_indices=True)

# --- Save JSONL ---
def save_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in dataset:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")

train_out = os.path.join(OUTPUT_DIR, "train.jsonl")
test_out = os.path.join(OUTPUT_DIR, "test.jsonl")

save_jsonl(train_dataset, train_out)
save_jsonl(test_dataset, test_out)

print(f"✅ Saved JSONL to {train_out} and {test_out}")