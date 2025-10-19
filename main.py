#!/usr/bin/env python3
"""
Main entry point for NSR (Negative Sample Reinforcement) training on Countdown task.

Based on: "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning"
Paper: https://arxiv.org/pdf/2506.01347
"""

import os
import datetime
import torch
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter

from nsr_countdown.config import TrainingConfig
from nsr_countdown.data import load_countdown_dataset, reward_fn
from nsr_countdown.model import init_policy, init_vllm, init_sampling_params
from nsr_countdown.trainer import train
from nsr_countdown.utils import get_constant_schedule_with_warmup, setup_logging

load_dotenv()
setup_logging()


def main() -> None:
    """Main training function."""
    # Load configuration
    config = TrainingConfig()
    
    # You can override config values here if needed:
    # config.n_grpo_steps = 100
    # config.objective = RLObjective.NSR
    # config.lambda_psr = 0.2
    
    print("=" * 80)
    print("NSR Training Configuration:")
    print(f"  Model: {config.model_id}")
    print(f"  RL Objective: {config.objective.value}")
    print(f"  Loss Type: {config.loss_type}")
    print(f"  Lambda (W-REINFORCE): {config.lambda_psr}")
    print(f"  Training Steps: {config.n_grpo_steps}")
    print(f"  Batch Size: {config.rollout_batch_size}")
    print(f"  Group Size: {config.group_size}")
    print("=" * 80)
    
    # Initialize policy and tokenizer
    print("\nInitializing policy model...")
    policy, tokenizer = init_policy(model_id=config.model_id, device=config.device)
    
    # Initialize vLLM engine
    print("Initializing vLLM engine...")
    llm = init_vllm(
        model_id=config.model_id,
        device=config.device,
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization
    )
    
    # Initialize sampling parameters
    sampling_params = init_sampling_params(
        temperature=config.temperature,
        min_tokens=config.min_tokens,
        max_tokens=config.max_tokens
    )
    
    # Load dataset
    print("Loading Countdown dataset...")
    train_examples = load_countdown_dataset("train", tokenizer, config.max_tokens)
    eval_examples = load_countdown_dataset("test", tokenizer, config.max_tokens)
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Eval examples: {len(eval_examples)}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0)
    
    # Setup TensorBoard logging
    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    log_dir = os.path.join(config.output_dir, "tb", f"nsr_{config.loss_type}", str(timestamp))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\nTensorBoard logs: {log_dir}")
    
    # Train
    print("\nStarting training...\n")
    train(
        policy=policy,
        tokenizer=tokenizer,
        llm=llm,
        sampling_params=sampling_params,
        train_prompts=[ex["prompt"] for ex in train_examples],
        train_answers=[ex["answer"] for ex in train_examples],
        eval_prompts=[ex["prompt"] for ex in eval_examples],
        eval_answers=[ex["answer"] for ex in eval_examples],
        optimizer=optimizer,
        scheduler=scheduler,
        n_grpo_steps=config.n_grpo_steps,
        rollout_batch_size=config.rollout_batch_size,
        group_size=config.group_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        clip_range=config.clip_range,
        use_std_normalization=config.use_std_normalization,
        advantage_eps=config.advantage_eps,
        device=config.device,
        eval_every=config.eval_every,
        writer=writer,
        seed=config.seed,
        loss_type=config.loss_type,
        max_completion_length=config.max_completion_length,
        objective=config.objective,
        lambda_psr=config.lambda_psr,
        reward_fn=reward_fn,
    )
    
    # Save model
    out_dir = os.path.join(config.output_dir, f"nsr_model_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\n{'=' * 80}")
    print(f"Training complete!")
    print(f"Model saved to: {out_dir}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"{'=' * 80}")
    
    writer.close()


if __name__ == "__main__":
    main()

