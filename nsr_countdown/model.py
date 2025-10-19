"""Model initialization: policy model and vLLM engine."""

import torch
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch


def init_policy(model_id: str, device: str) -> Tuple[PreTrainedModel, AutoTokenizer]:
    """
    Initialize the policy model and tokenizer.
    
    Args:
        model_id: Hugging Face model identifier.
        device: Device to load the model on.
    
    Returns:
        Tuple of (model, tokenizer).
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device).train()
    return model, tokenizer


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    """
    Initialize vLLM engine for fast rollout generation.
    
    Args:
        model_id: Hugging Face model identifier.
        device: Device for vLLM (typically "cuda").
        seed: Random seed for reproducibility.
        gpu_memory_utilization: Fraction of GPU memory to use.
    
    Returns:
        Initialized vLLM LLM instance.
    """
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """
    Load updated policy weights into the vLLM engine.
    
    Args:
        policy: The trained policy model.
        llm: The vLLM LLM instance.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def init_sampling_params(temperature: float, min_tokens: int, max_tokens: int) -> SamplingParams:
    """
    Initialize sampling parameters for vLLM generation.
    
    Args:
        temperature: Sampling temperature.
        min_tokens: Minimum tokens to generate.
        max_tokens: Maximum tokens to generate.
    
    Returns:
        SamplingParams configured for Countdown task.
    """
    sp = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        logprobs=0,
    )
    sp.stop = ["</answer>"]
    sp.include_stop_str_in_output = True
    return sp

