"""Configuration and hyperparameters for NSR training."""

from dataclasses import dataclass
from nsr_countdown.rl_objectives import RLObjective


@dataclass
class TrainingConfig:
    """Centralized configuration for NSR training on Countdown task."""
    
    # Model settings
    model_id: str = "Qwen/Qwen3-1.7B"
    device: str = "cuda"
    seed: int = 42
    gpu_memory_utilization: float = 0.4
    
    # Training hyperparameters
    n_grpo_steps: int = 200
    rollout_batch_size: int = 128
    group_size: int = 8
    gradient_accumulation_steps: int = 32
    learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    clip_range: float = 0.2
    advantage_eps: float = 1e-6
    
    # Sampling parameters
    temperature: float = 1.0
    min_tokens: int = 4
    max_tokens: int = 256
    
    # RL objective settings
    objective: RLObjective = RLObjective.W_REINFORCE
    lambda_psr: float = 0.1  # Weight for correct samples in W-REINFORCE
    
    # Loss type
    loss_type: str = "grpo"  # "grpo" or "dr_grpo"
    
    # Evaluation
    eval_every: int = 10
    
    # Logging
    output_dir: str = "./output"
    
    @property
    def use_std_normalization(self) -> bool:
        """Whether to use standard normalization (True for GRPO, False for DR-GRPO)."""
        return self.loss_type == "grpo"
    
    @property
    def max_completion_length(self) -> int:
        """Alias for max_tokens used in loss computation."""
        return self.max_tokens

