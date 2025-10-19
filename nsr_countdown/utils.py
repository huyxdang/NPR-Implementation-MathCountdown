"""Miscellaneous utility functions."""

import os
import logging
import warnings
from typing import List
from torch.optim.lr_scheduler import LambdaLR
import torch


def duplicate_data(arr: List, group_size: int) -> List:
    """
    Duplicate each element in the array `group_size` times.
    
    Args:
        arr: Input list.
        group_size: Number of times to duplicate each element.
    
    Returns:
        List with duplicated elements.
    """
    return [x for x in arr for _ in range(group_size)]


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a learning rate scheduler with linear warmup and constant learning rate.
    
    Args:
        optimizer: PyTorch optimizer.
        num_warmup_steps: Number of warmup steps.
        last_epoch: Last epoch number.
    
    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int):
        return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def setup_logging():
    """Configure logging to suppress verbose vLLM output."""
    logging.getLogger("vllm.engine.scheduler").setLevel(logging.ERROR)
    os.environ["VLLM_USE_V1"] = "0"
    warnings.filterwarnings("ignore")

