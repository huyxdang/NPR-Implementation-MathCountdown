"""Training loop orchestration for NSR."""

import random
import torch
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

from nsr_countdown.model import load_policy_into_vllm_instance
from nsr_countdown.tokenization import tokenize_prompt_and_output, get_response_log_probs
from nsr_countdown.rl_objectives import RLObjective, make_weighted_rewards
from nsr_countdown.rl_loss import (
    compute_group_normalized_advantages,
    compute_loss,
    masked_mean,
    masked_mean_drgrpo
)
from nsr_countdown.utils import duplicate_data
from nsr_countdown.evaluator import evaluate_model, log_eval, log_train


def rollout_with_vllm(
    policy: PreTrainedModel,
    llm: LLM,
    sampling_params: SamplingParams,
    prompts_batch: List[str],
    group_size: int
) -> Tuple[List[str], List[str], List[int]]:
    """
    Generate rollouts using vLLM for efficiency.
    
    Args:
        policy: Current policy model.
        llm: vLLM engine.
        sampling_params: Sampling configuration.
        prompts_batch: List of prompts to generate from.
        group_size: Number of completions per prompt.
    
    Returns:
        Tuple of (input_texts, response_texts, output_token_counts).
    """
    load_policy_into_vllm_instance(policy, llm)
    prompts_dup = duplicate_data(prompts_batch, group_size)
    vllm_rollouts = llm.generate(prompts_dup, sampling_params, use_tqdm=False)
    
    rollout_input_text, rollout_response_text, rollout_output_tokens = [], [], []
    for rollout in vllm_rollouts:
        for r in rollout.outputs:
            rollout_input_text.append(rollout.prompt)
            rollout_response_text.append(r.text)
            rollout_output_tokens.append(len(llm.llm_engine.tokenizer.encode(r.text)))
    
    return rollout_input_text, rollout_response_text, rollout_output_tokens


def grpo_microbatch_step(
    policy: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    advantages_per_seq: torch.Tensor,
    gradient_accumulation_steps: int,
    clip_range: float,
    loss_type: str = "grpo",
    max_completion_length: int = 512,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Perform a single microbatch gradient step.
    
    Args:
        policy: Policy model.
        input_ids: Input token IDs.
        labels: Target token IDs.
        response_mask: Mask for response tokens.
        advantages_per_seq: Advantage values per sequence.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        clip_range: PPO clipping range.
        loss_type: "grpo" or "dr_grpo".
        max_completion_length: Max completion length for DR-GRPO.
    
    Returns:
        Tuple of (loss, metadata).
    """
    policy_log_probs = get_response_log_probs(policy, input_ids, labels)
    old_log_probs = policy_log_probs.detach()
    advantages = advantages_per_seq.unsqueeze(-1)
    loss_per_token, metadata = compute_loss(advantages, policy_log_probs, old_log_probs, clip_range)
    
    if loss_type == "grpo":
        loss = masked_mean(loss_per_token, response_mask)
    elif loss_type == "dr_grpo":
        loss = masked_mean_drgrpo(loss_per_token, response_mask, max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata


def train(
    policy: PreTrainedModel,
    tokenizer: AutoTokenizer,
    llm: LLM,
    sampling_params: SamplingParams,
    *,
    train_prompts: List[str],
    train_answers: List[Dict],
    eval_prompts: List[str],
    eval_answers: List[Dict],
    optimizer: torch.optim.Optimizer,
    scheduler,
    n_grpo_steps: int,
    rollout_batch_size: int,
    group_size: int,
    gradient_accumulation_steps: int,
    clip_range: float,
    use_std_normalization: bool,
    advantage_eps: float,
    device: str,
    eval_every: int = 5,
    writer = None,
    seed: int,
    loss_type: str = "grpo",
    max_completion_length: int = 256,
    objective: RLObjective = RLObjective.RLVR,
    lambda_psr: float = 0.1,
    reward_fn = None,
) -> None:
    """
    Main training loop for NSR/GRPO.
    
    Args:
        policy: Policy model to train.
        tokenizer: Tokenizer.
        llm: vLLM engine for rollouts.
        sampling_params: Sampling configuration.
        train_prompts: Training prompts.
        train_answers: Training ground truths.
        eval_prompts: Evaluation prompts.
        eval_answers: Evaluation ground truths.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        n_grpo_steps: Number of training steps.
        rollout_batch_size: Batch size for rollouts.
        group_size: Group size for advantage computation.
        gradient_accumulation_steps: Gradient accumulation steps.
        clip_range: PPO clipping range.
        use_std_normalization: Whether to normalize advantages by std.
        advantage_eps: Epsilon for numerical stability.
        device: Device for training.
        eval_every: Evaluate every N steps.
        writer: TensorBoard writer.
        seed: Random seed.
        loss_type: "grpo" or "dr_grpo".
        max_completion_length: Max completion length.
        objective: RL objective (RLVR/PSR/NSR/W-REINFORCE).
        lambda_psr: Weight for correct samples in W-REINFORCE.
        reward_fn: Reward function.
    """
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    micro_train_batch_size = rollout_batch_size // gradient_accumulation_steps
    random.seed(seed)
    train_step = 0

    # Initial evaluation
    metrics = evaluate_model(llm, sampling_params, eval_prompts, eval_answers, reward_fn)
    if writer:
        for k in ["accuracy", "mean_reward", "std_reward", "avg_output_tokens", "count_correct", "count_partial", "count_failed"]:
            writer.add_scalar(f"eval/{k}", metrics[k], global_step=train_step)
        log_eval(metrics, writer, train_step)

    for _ in range(n_grpo_steps):
        # Sample training batch
        sampled = random.sample(list(zip(train_prompts, train_answers)), n_prompts_per_rollout_batch)
        prompts_batch, answers_batch = [p for p, _ in sampled], [a for _, a in sampled]
        
        # Generate rollouts
        rollout_input, rollout_response, rollout_tokens = rollout_with_vllm(
            policy, llm, sampling_params, prompts_batch, group_size
        )
        answers_dup = duplicate_data(answers_batch, group_size)
        avg_output_tokens = sum(rollout_tokens) / len(rollout_tokens) if rollout_tokens else 0.0
        
        # NSR/PSR filtering: Build weighted rewards + keep mask for objective
        weighted_rewards, keep_mask = make_weighted_rewards(
            rollout_response, answers_dup, reward_fn, objective=objective, lambda_psr=lambda_psr
        )

        # Calculate and log critical NSR metrics
        raw_rewards_for_metrics = torch.tensor(
            [reward_fn(r, gt) for r, gt in zip(rollout_response, answers_dup)],
            dtype=torch.float32
        )
        num_correct = (raw_rewards_for_metrics > 0).sum().item()
        correct_ratio = num_correct / len(raw_rewards_for_metrics)
        samples_kept = keep_mask.sum().item()
        
        if writer:
            writer.add_scalar("samples/correct_ratio", correct_ratio, train_step)
            writer.add_scalar("filtering/samples_kept", samples_kept, train_step)

        # If PSR/NSR filtered some samples, we must filter everything consistently
        if keep_mask.sum().item() != keep_mask.numel():
            keep_list = keep_mask.tolist()
            rollout_input = [t for t, k in zip(rollout_input, keep_list) if k]
            rollout_response = [t for t, k in zip(rollout_response, keep_list) if k]
            answers_dup = [t for t, k in zip(answers_dup, keep_list) if k]
            weighted_rewards = weighted_rewards[keep_mask]
            rollout_batch_size_effective = len(rollout_response)
        else:
            rollout_batch_size_effective = rollout_batch_size

        # Ensure length is a multiple of group_size (needed for .view(-1, group_size))
        rem = rollout_batch_size_effective % group_size
        if rem != 0:
            cut = rollout_batch_size_effective - rem
            rollout_input = rollout_input[:cut]
            rollout_response = rollout_response[:cut]
            answers_dup = answers_dup[:cut]
            weighted_rewards = weighted_rewards[:cut]
            rollout_batch_size_effective = cut
            if rollout_batch_size_effective == 0:
                # Skip this iteration if nothing remains
                continue

        # Compute advantages using precomputed (possibly weighted) rewards
        advantages, _, reward_meta = compute_group_normalized_advantages(
            rollout_response, answers_dup, reward_fn, group_size, advantage_eps, use_std_normalization,
            precomputed_rewards=weighted_rewards
        )

        # Tokenize the (possibly filtered) rollouts
        tokenized = tokenize_prompt_and_output(rollout_input, rollout_response, tokenizer)

        # Microbatch loop
        optimizer.zero_grad()
        rollout_loss = 0.0
        for micro_idx in range(0, rollout_batch_size_effective, micro_train_batch_size):
            s = slice(micro_idx, micro_idx + micro_train_batch_size)
            loss, _ = grpo_microbatch_step(
                policy,
                tokenized["input_ids"][s].to(device),
                tokenized["labels"][s].to(device),
                tokenized["response_mask"][s].to(device),
                advantages[s].to(device),
                gradient_accumulation_steps,
                clip_range,
                loss_type=loss_type,
                max_completion_length=max_completion_length
            )
            rollout_loss += float(loss.item())

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.grad is not None], 1.0
        )
        optimizer.step()
        scheduler.step()
        rollout_loss /= (rollout_batch_size_effective / micro_train_batch_size)
        train_step += 1
        
        print(f"Step {train_step} | Loss: {rollout_loss:.4f} | Grad: {grad_norm:.4f} | "
              f"Reward mean: {reward_meta['mean']:.4f} | Reward std: {reward_meta['std']:.4f} | "
              f"Correct: {correct_ratio:.2%} | Samples kept: {samples_kept}/{rollout_batch_size}")
        
        log_train(rollout_loss, grad_norm, reward_meta, avg_output_tokens, writer, train_step)
        
        if train_step % eval_every == 0:
            metrics = evaluate_model(llm, sampling_params, eval_prompts, eval_answers, reward_fn)
            log_eval(metrics, writer, train_step)

