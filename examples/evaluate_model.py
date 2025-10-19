#!/usr/bin/env python3
"""
Example: Evaluate a trained model on the Countdown task.

This shows how to use the modular components for evaluation only.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from nsr_countdown.data import load_countdown_dataset, reward_fn
from nsr_countdown.model import init_vllm, init_sampling_params
from nsr_countdown.evaluator import evaluate_model
from nsr_countdown.utils import setup_logging

setup_logging()


def evaluate_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """
    Evaluate a saved model checkpoint.
    
    Args:
        checkpoint_path: Path to saved model directory
        device: Device to run evaluation on
    """
    print("=" * 80)
    print(f"Evaluating model: {checkpoint_path}")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Initialize vLLM with the checkpoint
    print("Initializing vLLM...")
    llm = init_vllm(
        model_id=checkpoint_path,
        device=device,
        seed=42,
        gpu_memory_utilization=0.4
    )
    
    # Setup sampling
    sampling_params = init_sampling_params(
        temperature=1.0,
        min_tokens=4,
        max_tokens=256
    )
    
    # Load evaluation data
    print("Loading evaluation dataset...")
    eval_examples = load_countdown_dataset("test", tokenizer, max_tokens=256)
    
    # Evaluate
    print(f"\nEvaluating on {len(eval_examples)} examples...")
    metrics = evaluate_model(
        llm=llm,
        sampling_params=sampling_params,
        eval_prompts=[ex["prompt"] for ex in eval_examples],
        eval_answers=[ex["answer"] for ex in eval_examples],
        reward_fn=reward_fn
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Mean Reward: {metrics['mean_reward']:.4f}")
    print(f"Std Reward: {metrics['std_reward']:.4f}")
    print(f"Correct: {metrics['count_correct']}")
    print(f"Partial: {metrics['count_partial']}")
    print(f"Failed: {metrics['count_failed']}")
    print(f"Avg Output Tokens: {metrics['avg_output_tokens']:.1f}")
    print("=" * 80)
    
    # Show some example outputs
    examples = metrics['examples'][:5]
    print("\nExample Outputs:")
    print("-" * 80)
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"  Target: {ex['answer']['target']}")
        print(f"  Numbers: {ex['answer']['numbers']}")
        print(f"  Generated: {ex['equation']}")
        print(f"  Result: {ex['result']}")
        print(f"  Reward: {ex['reward']:.1f}")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "./output/nsr_model_<timestamp>"
    
    # Or use a base model to see baseline performance
    checkpoint_path = "Qwen/Qwen3-1.7B"
    
    print("\nNote: Update 'checkpoint_path' to point to your trained model.")
    print("Current path:", checkpoint_path)
    print()
    
    # Uncomment to run evaluation
    # evaluate_checkpoint(checkpoint_path)

