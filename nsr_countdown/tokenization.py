"""Tokenization utilities for training."""

import torch
import torch.nn.functional as F
from typing import Dict, List
from transformers import AutoTokenizer, PreTrainedModel


def tokenize_prompt_and_output(
    prompt_strs: List[str], 
    output_strs: List[str], 
    tokenizer: AutoTokenizer
) -> Dict[str, torch.Tensor]:
    """
    Tokenize prompt-output pairs for training.
    
    Args:
        prompt_strs: List of prompt strings.
        output_strs: List of output/response strings.
        tokenizer: Tokenizer instance.
    
    Returns:
        Dictionary with 'input_ids', 'labels', and 'response_mask' tensors.
    """
    batch_data = []
    max_len = 0
    
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer(prompt)["input_ids"]
        output_tokens = tokenizer(output)["input_ids"]
        combined_tokens = prompt_tokens + output_tokens
        max_len = max(max_len, len(combined_tokens))
        batch_data.append({
            "tokens": combined_tokens, 
            "prompt_len": len(prompt_tokens), 
            "total_len": len(combined_tokens)
        })
    
    batch_size = len(batch_data)
    input_ids = torch.full((batch_size, max_len - 1), tokenizer.eos_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len - 1), tokenizer.eos_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool)
    
    for i, data in enumerate(batch_data):
        tokens, seq_len = torch.tensor(data["tokens"]), len(data["tokens"])
        input_ids[i, :seq_len-1], labels[i, :seq_len-1] = tokens[:-1], tokens[1:]
        response_start, response_end = data["prompt_len"] - 1, seq_len - 1
        if response_end > response_start:
            response_mask[i, response_start:response_end] = True
    
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def get_response_log_probs(
    model: PreTrainedModel, 
    input_ids: torch.Tensor, 
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute log probabilities of generated tokens under the current policy.
    
    Args:
        model: The policy model.
        input_ids: Input token IDs [batch_size, seq_len].
        labels: Target token IDs [batch_size, seq_len].
    
    Returns:
        Log probabilities [batch_size, seq_len].
    """
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs

