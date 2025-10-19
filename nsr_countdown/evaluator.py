"""Evaluation and logging utilities."""

import torch
from typing import Any, Callable, Dict, List
from vllm import LLM, SamplingParams

from nsr_countdown.data import _extract_answer, _evaluate_equation


def evaluate_model(
    llm: LLM,
    sampling_params: SamplingParams,
    eval_prompts: List[str],
    eval_answers: List[Dict],
    reward_fn: Callable
) -> Dict[str, Any]:
    """
    Evaluate the model on a set of prompts.
    
    Args:
        llm: vLLM engine.
        sampling_params: Sampling configuration.
        eval_prompts: List of evaluation prompts.
        eval_answers: List of ground truth answers.
        reward_fn: Reward function to compute rewards.
    
    Returns:
        Dictionary with evaluation metrics and examples.
    """
    rollouts = llm.generate(eval_prompts, sampling_params)
    examples, rewards, output_token_lengths = [], [], []
    
    for rollout, gt in zip(rollouts, eval_answers):
        response_text = rollout.outputs[0].text
        reward_value = reward_fn(response_text, gt)
        equation = _extract_answer(response_text)
        result = _evaluate_equation(equation) if equation is not None else None
        output_tokens = len(llm.llm_engine.tokenizer.encode(response_text))
        output_token_lengths.append(output_tokens)
        
        examples.append({
            "prompt": rollout.prompt,
            "response": response_text,
            "answer": gt,
            "equation": equation,
            "result": result,
            "reward": reward_value,
            "output_tokens": output_tokens,
        })
        rewards.append(reward_value)
    
    rewards_tensor = torch.tensor(rewards) if rewards else torch.tensor([0.0])
    tol = 1e-8
    count_correct = sum(1 for r in rewards if abs(r - 1.0) < tol)
    count_partial = sum(1 for r in rewards if abs(r - 0.1) < tol)
    count_failed = sum(1 for r in rewards if abs(r - 0.0) < tol)
    accuracy = (count_correct / len(rewards)) * 100 if rewards else 0.0
    avg_output_tokens = sum(output_token_lengths) / len(output_token_lengths) if output_token_lengths else 0.0
    
    return {
        "mean_reward": float(rewards_tensor.mean().item()),
        "std_reward": float(rewards_tensor.std().item()) if rewards_tensor.numel() > 1 else 0.0,
        "num_examples": len(rewards),
        "examples": examples,
        "count_correct": count_correct,
        "count_partial": count_partial,
        "count_failed": count_failed,
        "accuracy": accuracy,
        "avg_output_tokens": avg_output_tokens,
    }


def _format_eval_example(example: Dict[str, Any]) -> str:
    """Format a single evaluation example for logging."""
    target = example["answer"]["target"] if isinstance(example.get("answer"), dict) and "target" in example["answer"] else "?"
    numbers = example["answer"].get("numbers") if isinstance(example.get("answer"), dict) else None
    return (
        f"Prompt: {example.get('prompt', '')}\n"
        f"Response: {example.get('response', '')}\n"
        f"Equation: {example.get('equation', None)} | Result: {example.get('result', None)} | Target: {target} | Numbers: {numbers}\n"
        f"Reward: {example.get('reward', 0.0):.3f}\n"
    )


def log_train(
    rollout_batch_loss: float,
    grad_norm: float,
    reward_metadata: Dict[str, Any],
    avg_output_tokens: float,
    writer,
    step: int
) -> None:
    """
    Log training metrics to TensorBoard.
    
    Args:
        rollout_batch_loss: Batch loss value.
        grad_norm: Gradient norm.
        reward_metadata: Dictionary with reward statistics.
        avg_output_tokens: Average output tokens.
        writer: TensorBoard writer.
        step: Current training step.
    """
    if writer:
        writer.add_scalar("train/loss", float(rollout_batch_loss), global_step=step)
        writer.add_scalar("train/grad_norm", float(grad_norm), global_step=step)
        writer.add_scalar("train/reward_mean", float(reward_metadata["mean"]), global_step=step)
        writer.add_scalar("train/reward_std", float(reward_metadata["std"]), global_step=step)
        writer.add_scalar("train/avg_output_tokens", float(avg_output_tokens), global_step=step)
    
    print(f"Step {step} | Loss: {rollout_batch_loss:.4f} | Grad norm: {grad_norm:.4f} | "
          f"Reward mean: {float(reward_metadata['mean']):.4f} | Reward std: {float(reward_metadata['std']):.4f} | "
          f"Avg output tokens: {avg_output_tokens:.1f}")


def log_eval(metrics: Dict[str, Any], writer, step: int) -> None:
    """
    Log evaluation metrics and examples to TensorBoard.
    
    Args:
        metrics: Dictionary with evaluation metrics.
        writer: TensorBoard writer.
        step: Current training step.
    """
    examples = metrics.get("examples", []) or []
    if not examples:
        return
    
    tol = 1e-8
    correct_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 1.0) < tol][:10]
    partial_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 0.1) < tol][:10]
    failed_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 0.0) < tol][:10]
    
    if correct_examples:
        print(f"\n=== Eval examples (CORRECT, reward=1.0) @ step {step} ===")
        for idx, ex in enumerate(correct_examples[:2], 1):
            print(f"[CORRECT #{idx}]\n" + _format_eval_example(ex))
    
    if partial_examples:
        print(f"\n=== Eval examples (PARTIAL, reward=0.1) @ step {step} ===")
        for idx, ex in enumerate(partial_examples[:2], 1):
            print(f"[PARTIAL #{idx}]\n" + _format_eval_example(ex))
    
    if failed_examples:
        print(f"\n=== Eval examples (FAILED, reward=0.0) @ step {step} ===")
        for idx, ex in enumerate(failed_examples[:2], 1):
            print(f"[FAILED #{idx}]\n" + _format_eval_example(ex))
    
    if writer:
        correct_text = "\n\n".join([_format_eval_example(ex) for ex in correct_examples]) or ""
        partial_text = "\n\n".join([_format_eval_example(ex) for ex in partial_examples]) or ""
        failed_text = "\n\n".join([_format_eval_example(ex) for ex in failed_examples]) or ""
        if correct_text:
            writer.add_text("eval/examples_correct", correct_text, global_step=step)
        if partial_text:
            writer.add_text("eval/examples_partial", partial_text, global_step=step)
        if failed_text:
            writer.add_text("eval/examples_failed", failed_text, global_step=step)
        
        writer.add_scalar("eval/accuracy", metrics["accuracy"], global_step=step)
        writer.add_scalar("eval/mean_reward", metrics["mean_reward"], global_step=step)
        writer.add_scalar("eval/std_reward", metrics["std_reward"], global_step=step)
        writer.add_scalar("eval/avg_output_tokens", metrics["avg_output_tokens"], global_step=step)
    
    print(f"Eval @ step {step}: accuracy={metrics['accuracy']:.1f}% mean_reward={metrics['mean_reward']:.4f} "
          f"avg_tokens={metrics['avg_output_tokens']:.1f} | correct:{metrics['count_correct']} "
          f"partial:{metrics['count_partial']} failed:{metrics['count_failed']}")

