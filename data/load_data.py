from datasets import Dataset, DatasetDict
import os
import json

# --- CONFIG ---
DATA_ROOT = "./MATH"   # adjust this to where your local MATH folder is
HF_REPO = "huyxdang/MATH-dataset"  # change to your HF username/repo

def load_split(split_dir):
    """Recursively read all JSONs and return list of dicts with 4 keys."""
    data = []
    for subdir, _, files in os.walk(split_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(subdir, file)
                with open(path, "r") as f:
                    raw = json.load(f)
                    # normalize fields
                    item = {
                        "problem": raw.get("problem", ""),
                        "solution": raw.get("solution", ""),
                        "level": raw.get("level", ""),
                        "type": raw.get("type", ""),
                    }
                    data.append(item)
    return data

# --- LOAD TRAIN/TEST ---
train_data = load_split(os.path.join(DATA_ROOT, "train"))
test_data = load_split(os.path.join(DATA_ROOT, "test"))

print(f"Loaded {len(train_data)} train and {len(test_data)} test samples.")

# --- CREATE HF DATASET ---
train_ds = Dataset.from_list(train_data)
test_ds = Dataset.from_list(test_data)
dataset = DatasetDict({"train": train_ds, "test": test_ds})

# --- UPLOAD TO HUGGING FACE ---
dataset.push_to_hub(HF_REPO)
print(f"✅ Uploaded to Hugging Face: https://huggingface.co/datasets/{HF_REPO}")
