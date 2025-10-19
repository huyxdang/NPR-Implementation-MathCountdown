# data.py

# Pipeline to process HF dataset

from datasets import load_dataset

def load_countdown(split="train"):
    samples = []
    train_dataset = load_dataset("https://huggingface.co/datasets/justinphan3110/Countdown-Tasks-3to4", split = split)
    for ex in train_dataset: 
        target = ex["target"]
        nums = ex["nums"]
        prompt = f"Using numbers {nums}, make {target}. 
        You must use all numbers, each number only once, 
        and with operations + - / * only.
        Do reasoning steps inside <think> </think> and put your final answer in <answer> </answer>."
        samples.append({
            prompt: prompt,
            nums: nums,
            target: target,
        })
    return samples