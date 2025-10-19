"""RL loss computation: advantages, PPO loss, and masked averaging."""

import torch
from typing import Callable, Dict, List, Tuple


def compute_group_normalized_advantages(
    rollout_responses: List[str],
    repeated_ground_truths: List[Dict],
    reward_fn: Callable[[str, Dict], float],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
    precomputed_rewards: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute group-normalized advantages for GRPO.
    
    Groups responses together and normalizes advantages within each group,
    which reduces variance and improves training stability.
    
    Args:
        rollout_responses: List of generated responses.
        repeated_ground_truths: Corresponding ground truth data.
        reward_fn: Function to compute rewards.
        group_size: Number of samples per group.
        advantage_eps: Epsilon for numerical stability in normalization.
        normalize_by_std: Whether to normalize by standard deviation.
        precomputed_rewards: Optional pre-weighted rewards (for NSR/PSR/W-REINFORCE).
    
    Returns:
        advantages: Tensor of advantages [N].
        raw_rewards: Tensor of raw rewards [N].
        metadata: Dictionary with reward statistics.
    """
    # 1) Compute or use provided rewards
    if precomputed_rewards is None:
        rewards = [reward_fn(resp, gt) for resp, gt in zip(rollout_responses, repeated_ground_truths)]
        raw_rewards = torch.tensor(rewards, dtype=torch.float32)
    else:
        raw_rewards = precomputed_rewards.float()

    # 2) Reshape into groups (assumes length is multiple of group_size)
    rewards_grouped = raw_rewards.view(-1, group_size)
    group_mean = rewards_grouped.mean(dim=1, keepdim=True)
    group_std = rewards_grouped.std(dim=1, keepdim=True, unbiased=False)

    advantages = rewards_grouped - group_mean
    if normalize_by_std:
        advantages = advantages / (group_std + advantage_eps)

    advantages = advantages.flatten()
    metadata = {
        "mean": torch.mean(raw_rewards),
        "std": torch.std(raw_rewards, unbiased=False),
        "max": torch.max(raw_rewards),
        "min": torch.min(raw_rewards),
    }
    return advantages, raw_rewards, metadata


def compute_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_range: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the per-token PPO clipped surrogate loss.
    
    This implements the PPO clipping mechanism to prevent overly large policy updates.
    
    Args:
        advantages: Advantage values [batch_size, seq_len].
        policy_log_probs: Log probs under current policy [batch_size, seq_len].
        old_log_probs: Log probs under old policy [batch_size, seq_len].
        clip_range: Clipping range for policy ratio (typically 0.2).
    
    Returns:
        loss: Per-token loss [batch_size, seq_len].
        stats: Dictionary with ratio statistics for logging.
    """
    # 1. Ratio between new and old policies
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)

    # 2. Unclipped term
    unclipped = advantages * pi_ratio

    # 3. Clipped term
    clipped_ratio = torch.clamp(pi_ratio, 1 - clip_range, 1 + clip_range)
    clipped = advantages * clipped_ratio

    # 4. Take elementwise minimum and negate (PPO-style objective)
    loss = -torch.minimum(unclipped, clipped)

    # Optional metadata for logging/debugging
    stats = {
        "ratio_mean": pi_ratio.mean(),
        "ratio_std": pi_ratio.std(),
        "ratio_min": pi_ratio.min(),
        "ratio_max": pi_ratio.max(),
    }
    
    return loss, stats


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean of tensor values where mask=True for each row, then average across the batch.
    
    This is used for GRPO loss averaging, where we only average over response tokens.
    
    Args:
        tensor: Tensor to average [batch_size, seq_len].
        mask: Boolean mask indicating which tokens to include [batch_size, seq_len].
    
    Returns:
        Scalar loss value.
    """
    # Convert mask to float for proper multiplication
    mask = mask.float()
    
    # Sum over tokens for each sequence, considering only response tokens
    masked_sum = (tensor * mask).sum(dim=1)
    
    # Count valid tokens per sequence
    token_count = mask.sum(dim=1).clamp(min=1)
    
    # Mean over valid tokens in each sequence, then average across batch
    mean_per_seq = masked_sum / token_count
    loss = mean_per_seq.mean()
    
    return loss


def masked_mean_drgrpo(tensor: torch.Tensor, mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """
    Compute the sum of tensor values where mask=True, divided by num_tokens, then average across the batch.
    
    This is used for the DR-GRPO (Dense Reward GRPO) loss variant.
    
    Args:
        tensor: Tensor to average [batch_size, seq_len].
        mask: Boolean mask indicating which tokens to include [batch_size, seq_len].
        num_tokens: Fixed token count for normalization.
    
    Returns:
        Scalar loss value.
    """
    # Convert mask to float for proper multiplication
    mask = mask.float()
    
    masked_sum = (tensor * mask).sum(dim=1)
    
    # Divide by fixed constant num_tokens (same for all sequences)
    mean_per_seq = masked_sum / float(num_tokens)
    loss = mean_per_seq.mean()
    
    return loss

