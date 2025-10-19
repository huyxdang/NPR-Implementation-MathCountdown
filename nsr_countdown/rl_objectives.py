"""RL objective definitions: RLVR, PSR, NSR, and W-REINFORCE."""

import torch
from enum import Enum
from typing import Callable, Dict, List, Tuple


class RLObjective(str, Enum):
    """Reinforcement learning objectives for training."""
    RLVR = "rlvr"            # Use all samples, rewards = {+1, -1}
    PSR = "psr"              # Keep only correct samples (Positive Sample Reinforcement)
    NSR = "nsr"              # Keep only incorrect samples (Negative Sample Reinforcement)
    W_REINFORCE = "w_reinforce"  # Weighted REINFORCE: rewards = {+λ, -1}


def make_weighted_rewards(
    rollout_responses: List[str],
    repeated_ground_truths: List[Dict],
    base_reward_fn: Callable[[str, Dict], float],
    objective: RLObjective,
    lambda_psr: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply objective-specific reward weighting and sample filtering.
    
    This is the key function that implements the different RL objectives from the paper:
    - RLVR: Standard RL with all samples
    - PSR: Train only on correct samples
    - NSR: Train only on incorrect samples (negative reinforcement)
    - W-REINFORCE: Weighted rewards (+λ for correct, -1 for incorrect)
    
    Args:
        rollout_responses: List of generated responses.
        repeated_ground_truths: Corresponding ground truth data.
        base_reward_fn: Function to compute binary rewards {+1, -1}.
        objective: Which RL objective to use.
        lambda_psr: Weight for correct samples in W-REINFORCE (typically 0.1).
    
    Returns:
        rewards: torch.FloatTensor [N] - Weighted rewards.
        keep_mask: torch.BoolTensor [N] - Which samples to keep for training.
    """
    raw = [base_reward_fn(r, gt) for r, gt in zip(rollout_responses, repeated_ground_truths)]
    rewards = []
    keep = []
    
    for rv in raw:
        if objective == RLObjective.RLVR:
            # Standard: use all samples with binary rewards
            rewards.append(rv)  # +1 or -1
            keep.append(True)
            
        elif objective == RLObjective.PSR:
            # Positive Sample Reinforcement: only train on correct samples
            keep.append(rv > 0)
            if rv > 0:
                rewards.append(1.0)
                
        elif objective == RLObjective.NSR:
            # Negative Sample Reinforcement: only train on incorrect samples
            keep.append(rv < 0)
            if rv < 0:
                rewards.append(-1.0)
                
        elif objective == RLObjective.W_REINFORCE:
            # Weighted REINFORCE: +λ for correct, -1 for incorrect
            if rv > 0:
                rewards.append(lambda_psr)
                keep.append(True)
            else:
                rewards.append(-1.0)
                keep.append(True)
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    rewards = torch.tensor(rewards, dtype=torch.float32)
    keep_mask = torch.tensor(keep, dtype=torch.bool)
    return rewards, keep_mask

