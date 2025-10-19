#!/usr/bin/env python3
"""
Example: Custom training configuration with different RL objectives.

This demonstrates how to use the modular structure to run experiments
with different training configurations.
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
from nsr_countdown.rl_objectives import RLObjective
from nsr_countdown.utils import get_constant_schedule_with_warmup, setup_logging

load_dotenv()
setup_logging()


def experiment_nsr_only():
    """Experiment: Train only on incorrect samples (NSR)."""
    print("\n" + "=" * 80)
    print("EXPERIMENT: NSR (Negative Sample Reinforcement)")
    print("Training only on incorrect samples")
    print("=" * 80 + "\n")
    
    config = TrainingConfig()
    config.objective = RLObjective.NSR
    config.n_grpo_steps = 100  # Shorter for demo
    
    return run_experiment(config, "nsr_experiment")


def experiment_psr_only():
    """Experiment: Train only on correct samples (PSR)."""
    print("\n" + "=" * 80)
    print("EXPERIMENT: PSR (Positive Sample Reinforcement)")
    print("Training only on correct samples")
    print("=" * 80 + "\n")
    
    config = TrainingConfig()
    config.objective = RLObjective.PSR
    config.n_grpo_steps = 100
    
    return run_experiment(config, "psr_experiment")


def experiment_w_reinforce():
    """Experiment: Weighted REINFORCE (paper's best method)."""
    print("\n" + "=" * 80)
    print("EXPERIMENT: W-REINFORCE (Weighted REINFORCE)")
    print("Lambda = 0.1 (paper's recommended value)")
    print("=" * 80 + "\n")
    
    config = TrainingConfig()
    config.objective = RLObjective.W_REINFORCE
    config.lambda_psr = 0.1
    config.n_grpo_steps = 100
    
    return run_experiment(config, "w_reinforce_experiment")


def experiment_dr_grpo():
    """Experiment: Using DR-GRPO loss variant."""
    print("\n" + "=" * 80)
    print("EXPERIMENT: DR-GRPO Loss Variant")
    print("Dense Reward GRPO with W-REINFORCE")
    print("=" * 80 + "\n")
    
    config = TrainingConfig()
    config.objective = RLObjective.W_REINFORCE
    config.lambda_psr = 0.1
    config.loss_type = "dr_grpo"
    config.n_grpo_steps = 100
    
    return run_experiment(config, "dr_grpo_experiment")


def run_experiment(config: TrainingConfig, experiment_name: str):
    """
    Run a training experiment with the given configuration.
    
    Args:
        config: Training configuration
        experiment_name: Name for logging and outputs
    """
    # Initialize model
    print(f"Initializing {config.model_id}...")
    policy, tokenizer = init_policy(model_id=config.model_id, device=config.device)
    
    # Initialize vLLM
    print("Initializing vLLM engine...")
    llm = init_vllm(
        model_id=config.model_id,
        device=config.device,
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization
    )
    
    # Setup sampling
    sampling_params = init_sampling_params(
        temperature=config.temperature,
        min_tokens=config.min_tokens,
        max_tokens=config.max_tokens
    )
    
    # Load data
    print("Loading dataset...")
    train_examples = load_countdown_dataset("train", tokenizer, config.max_tokens)
    eval_examples = load_countdown_dataset("test", tokenizer, config.max_tokens)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0)
    
    # Setup logging
    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    log_dir = os.path.join(config.output_dir, "tb", experiment_name, str(timestamp))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"\nStarting training for {experiment_name}...")
    print(f"TensorBoard: {log_dir}\n")
    
    # Train
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
    out_dir = os.path.join(config.output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    
    print(f"\n{'=' * 80}")
    print(f"Experiment '{experiment_name}' complete!")
    print(f"Model: {out_dir}")
    print(f"Logs: {log_dir}")
    print(f"{'=' * 80}\n")
    
    writer.close()
    return out_dir


if __name__ == "__main__":
    # Run different experiments
    # Uncomment the one you want to run:
    
    # experiment_nsr_only()
    # experiment_psr_only()
    experiment_w_reinforce()  # Paper's best method
    # experiment_dr_grpo()

