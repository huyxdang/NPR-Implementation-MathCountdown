# NSR (Negative Sample Reinforcement) on MATH Dataset

## Overview

This repository implements and investigates the effectiveness of Negative Sample Reinforcement (NSR), Weighted REINFORCE (W-REINFORCE), and Positive Sample Reinforcement (PSR) objectives on mathematical reasoning tasks using the MATH dataset. The work is an extension of the paper "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning" (Zhu et al., 2025) and explores how these reinforcement learning objectives perform across different model sizes and capabilitie (priors).

## Research Question

**How do NSR and W-REINFORCE objectives change with the model's prior knowledge and capabilities?**

We investigate whether smaller models with less prior mathematical knowledge benefit differently from negative reinforcement compared to larger, more capable models. This study examines the relationship between model size, mathematical reasoning ability, and the effectiveness of different RL objectives.

## Models Used

We conduct experiments across the Qwen2.5 instruction-tuned model family:

- **Qwen2.5-0.5B-Instruct** (0.49B parameters)
- **Qwen2.5-1.5B-Instruct** (1.5B parameters) 
- **Qwen2.5-3B-Instruct** (3B parameters)

These models provide a good range of capabilities while maintaining the same architecture and training methodology, allowing for controlled comparison of how model priors may affect RL objective effectiveness.

## Experimental Configuration

### Training Parameters
- **Max Tokens**: 512
- **Training Steps**: 80
- **Rollout Batch Size**: 64
- **Mini-batch Size**: 32 (64/2 = 32)
- **Temperature**: 1.0 (consistent across all experiments)
- **Rollouts per Prompt**: 2
- **Learning Rate**: 1e-6
- **Gradient Accumulation Steps**: 2
- **PPO Clip Range**: 0.2

We couldn't replicate the paper's parameters due to a lack of compute. 

### Dataset
- **Source**: MATH dataset (Hendrycks et al.)
- **Processing**: Stratified sampling (25% subset) maintaining original train/test distributions
- **Format**: Problems with step-by-step solutions and `\boxed{}` final answers
- **Evaluation**: Answer verification using `math-verify` library

### RL Objectives
1. **NSR (Negative Sample Reinforcement)**: Trains only on incorrect samples with -1.0 reward
2. **PSR (Positive Sample Reinforcement)**: Trains only on correct samples with +1.0 reward  
3. **W-REINFORCE**: Weighted approach with +λ for correct, -1.0 for incorrect samples

## Results

### Training Curves

The following visualization shows the training progress across different model sizes and RL objectives:

![Training Curves](visualizations/training_curves_smoothed.png) (Smoothed via MA with Window = 2)

### Final Performance Comparison

The bar chart below compares the final performance of different RL objectives across model sizes:

![Final Performance Comparison](visualizations/final_performance_comparison.png)

### Analysis


## Usage

### Running Experiments

```bash
# NSR with 0.5B model
python MATH_experiment.py --objective NSR --model_id Qwen/Qwen2.5-0.5B-Instruct

# W-REINFORCE with 3B model  
python MATH_experiment.py --objective W_REINFORCE --model_id Qwen/Qwen2.5-3B-Instruct

# PSR with 1.5B model
python MATH_experiment.py --objective PSR --model_id Qwen/Qwen2.5-1.5B-Instruct
```

## Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- vLLM >= 0.2.0
- math-verify >= 0.1.0
- Datasets >= 2.14.0
- TensorBoard >= 2.14.0


## Acknowledgments

- Inspired by "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning"
- Built on the MATH dataset by Hendrycks et al.
- Uses Qwen2.5 models by Alibaba Cloud
