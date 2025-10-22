# NSR (Negative Sample Reinforcement) on MATH Dataset

## Overview

This repository implements and investigates the effectiveness of Negative Sample Reinforcement (NSR) and Weighted REINFORCE (W-REINFORCE) objectives on mathematical reasoning tasks using the MATH dataset. The work is inspired by the paper "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning" and explores how these reinforcement learning objectives perform across different model sizes and capabilities.

## Research Question

**How do NSR and W-REINFORCE objectives change with the model's prior knowledge and capabilities?**

We investigate whether smaller models with less prior mathematical knowledge benefit differently from negative reinforcement compared to larger, more capable models. This study examines the relationship between model size, mathematical reasoning ability, and the effectiveness of different RL objectives.

## Models Used

We conduct experiments across the Qwen2.5 instruction-tuned model family:

- **Qwen2.5-0.5B-Instruct** (0.49B parameters)
- **Qwen2.5-1.5B-Instruct** (1.5B parameters) 
- **Qwen2.5-3B-Instruct** (3B parameters)

These models provide a good range of capabilities while maintaining the same architecture and training methodology, allowing for controlled comparison of how model size affects RL objective effectiveness.

## Experimental Configuration

### Training Parameters
- **Max Tokens**: 8,192 (increased from default to allow for detailed mathematical reasoning)
- **Training Steps**: 100
- **Rollout Batch Size**: 2,048
- **Mini-batch Size**: 256
- **Temperature**: 1.0 (consistent across all experiments)
- **Rollouts per Prompt**: 8
- **Learning Rate**: 1e-6
- **Gradient Accumulation Steps**: 8
- **PPO Clip Range**: 0.2

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

*Results section to be updated after experiment completion*

### Expected Analysis
- **Model Size vs. Objective Effectiveness**: How does the optimal RL objective change with model capability?
- **Learning Dynamics**: Do smaller models require different reward structures?
- **Mathematical Reasoning Improvement**: Quantitative analysis of accuracy improvements across different problem types
- **Convergence Patterns**: How quickly do different objectives converge for different model sizes?

## Usage

### Running Experiments

```bash
# NSR with 0.5B model
python MATH_experiment.py --objective NSR --model_id Qwen/Qwen2.5-0.5B-Instruct

# W-REINFORCE with 3B model  
python MATH_experiment.py --objective W_REINFORCE --model_id Qwen/Qwen2.5-3B-Instruct --lambda_psr 0.1

# PSR with 1.5B model
python MATH_experiment.py --objective PSR --model_id Qwen/Qwen2.5-1.5B-Instruct
```

### Data Preprocessing

```bash
# Process MATH dataset with stratified sampling
python data/pre_process.py
```

### Testing Utilities

```bash
# Test answer extraction and verification
python test_utils.py
```

## Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- vLLM >= 0.2.0
- math-verify >= 0.1.0
- Datasets >= 2.14.0
- TensorBoard >= 2.14.0

## File Structure

```
├── MATH_experiment.py          # Main experiment script
├── utils.py                    # Answer extraction and verification utilities
├── data/
│   ├── pre_process.py         # Dataset preprocessing with stratified sampling
│   └── math_json/             # Processed MATH dataset (JSONL format)
├── test_utils.py              # Unit tests for utilities
└── requirements.txt           # Python dependencies
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{nsr-math-experiments,
  title={NSR and W-REINFORCE on MATH Dataset: Investigating Model Size Effects},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/NPR-Implementation-MathCountdown-1}
}
```

## Acknowledgments

- Inspired by "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning"
- Built on the MATH dataset by Hendrycks et al.
- Uses Qwen2.5 models by Alibaba Cloud
