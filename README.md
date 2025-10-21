# Negative Sample Reinforcement for Math Countdown

[![Paper](https://img.shields.io/badge/arXiv-2506.01347-b31b1b.svg)](https://arxiv.org/pdf/2506.01347)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of **"The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning"** applied to the Math Countdown task. This work explores training language models with different sample selection strategies: learning from mistakes (NSR), learning from successes (PSR), and weighted combinations (W-REINFORCE).

## Math Countdown Task

The Countdown task requires generating valid arithmetic equations using a given set of numbers to reach a target value. Each number must be used exactly once.

**Example:**
```
Input:  numbers = [1, 2, 3, 4], target = 5
Output: (1 + 2) * 3 - 4 = 5 ✓
```

**Evaluation Criteria:**
- Equation must contain a valid `<answer>...</answer>` tag
- All and only the given numbers are used
- Arithmetic evaluation equals the target value

**Dataset:** `justinphan3110/Countdown-Tasks-3to4` (3-4 digit problems)

## Paper Summary

This implementation compares four reinforcement learning objectives for training language models on verifiable reasoning tasks:

| Objective | Training Samples | Reward Weighting | Description |
|-----------|-----------------|------------------|-------------|
| **RLVR** | All samples | ±1 | Standard RL baseline |
| **PSR** | Correct only | +1 | Positive Sample Reinforcement |
| **NSR** | Incorrect only | -1 | Negative Sample Reinforcement (key contribution) |
| **W-REINFORCE** | All samples | +λ (correct), -1 (incorrect) | Weighted combination (paper's best) |

**Key Finding:** Training on mistakes (NSR) and weighted objectives (W-REINFORCE with λ=0.1-0.3) can outperform training only on correct samples, challenging conventional imitation learning approaches.

## Results

Performance on the Countdown test set (accuracy %):

| Method | Max Tokens | Accuracy (%) | Notes |
|--------|-----------|--------------|-------|
| **Baseline (Zero-shot)** | 256 | _TBD_ | No fine-tuning |
| **Baseline (Zero-shot)** | 512 | _TBD_ | No fine-tuning |
| **NSR** | 256 | _TBD_ | Train on incorrect only |
| **NSR** | 512 | _TBD_ | Train on incorrect only |
| **PSR** | 256 | _TBD_ | Train on correct only |
| **PSR** | 512 | _TBD_ | Train on correct only |
| **W-REINFORCE (λ=0.3)** | 256 | _TBD_ | Weighted combination |
| **W-REINFORCE (λ=0.3)** | 512 | _TBD_ | Weighted combination |

All experiments use 200 training steps on Qwen3-1.7B with identical hyperparameters (see below).

## Installation

```bash
# Clone repository
git clone <repository-url>
cd NPR-Implementation-MathCountdown-1

# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention for faster training
pip install flash-attn --no-build-isolation
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended: 24GB+ VRAM)
- vLLM 0.4.0+

## Quick Start

### Run Baseline Evaluation (Zero-shot)

```bash
# Evaluate pre-trained model without fine-tuning
python baseline.py --max_tokens 256
python baseline.py --max_tokens 512
python baseline.py --max_tokens 1024
```

### Run All Experiments

```bash
# Run all 9 fine-tuning experiments sequentially
bash run_experiments.sh
```

### Full Experiment Suite

```bash
# Experiment 1: NSR with 256 tokens
python experiment.py --objective NSR --max_tokens 256

# Experiment 2: NSR with 512 tokens
python experiment.py --objective NSR --max_tokens 512

# Experiment 3: NSR with 1024 tokens
python experiment.py --objective NSR --max_tokens 1024

# Experiment 4: PSR with 256 tokens
python experiment.py --objective PSR --max_tokens 256

# Experiment 5: PSR with 512 tokens
python experiment.py --objective PSR --max_tokens 512

# Experiment 6: NSR with 1024 tokens
python experiment.py --objective PSR --max_tokens 1024

# Experiment 7: W-REINFORCE (λ=0.1) with 256 tokens
python experiment.py --objective W_REINFORCE --max_tokens 256 

# Experiment 8: W-REINFORCE (λ=0.1) with 512 tokens
python experiment.py --objective W_REINFORCE --max_tokens 512 

# Experiment 9: W-REINFORCE (λ=0.1) with 1024 tokens
python experiment.py --objective W_REINFORCE --max_tokens 1024 
```

## Hyperparameters

Complete hyperparameter configuration (all experiments use identical settings except objective and max_tokens):

### Core Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_id` | `Qwen/Qwen3-1.7B` | Pre-trained model |
| `--n_train_steps` | `200` | Number of training iterations |
| `--lr` | `3e-6` | Learning rate (AdamW) |
| `--seed` | `42` | Random seed for reproducibility |
| `--device` | `cuda` | Training device |

### Rollout & Batch Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rollout_batch_size` | `128` | Total samples per training step |
| `--group_size` | `8` | Samples per prompt (for advantage normalization) |
| `--grad_acc_steps` | `16` | Gradient accumulation steps |
| `--temperature` | `0.7` | Sampling temperature |
| `--max_tokens` | `256` | Max response length (variable: 256 or 512) |

### RL Objective Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--objective` | `NSR` | RL objective: `RLVR`, `PSR`, `NSR`, or `W_REINFORCE` |
| `--lambda_psr` | `0.1` | Weight for correct samples (W-REINFORCE only) |
| `--clip_range` | `0.2` | PPO clipping parameter |

### System Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpu_mem_util` | `0.4` | vLLM GPU memory utilization (0-1) |
| `--eval_every` | `10` | Evaluation frequency (steps) |
| `--loss_type` | `standard` | Loss computation: `standard` or `length_normalized` |
| `--output_dir` | `./output` | Directory for checkpoints and logs |

### Fixed Parameters (Not Configurable via CLI)

- Optimizer: AdamW (weight_decay=1e-2, betas=(0.9, 0.95))
- Advantage epsilon: 1e-4
- Minimum tokens: 4
- Learning rate schedule: Constant (no warmup)
- Prompt template: Countdown task with `<think>` and `<answer>` tags

## Repository Structure

```
.
├── experiment.py           # Main training script
├── baseline.py             # Zero-shot evaluation script
├── run_experiments.sh      # Launcher for all 6 experiments
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Implementation Details

### Reward Function

Binary verifiable rewards based on correctness:

```python
def reward_fn(response, ground_truth):
    equation = extract_answer(response)
    if not equation:
        return -1.0
    if not validate_numbers(equation, ground_truth["numbers"]):
        return -1.0
    if not evaluate_equation(equation) == ground_truth["target"]:
        return -1.0
    return +1.0
```

### Group-Normalized Advantages

Advantages computed using group normalization (GRPO):

```
advantages = (rewards - group_mean) / (group_std + ε)
```

where groups contain `group_size=8` rollouts per prompt.

### Sample Filtering

Objective-specific filtering in `make_weighted_rewards()`:

- **RLVR**: Keep all samples, rewards ∈ {-1, +1}
- **PSR**: Keep only rewards > 0, set to +1
- **NSR**: Keep only rewards < 0, set to -1
- **W-REINFORCE**: Keep all samples, rewards ∈ {-1, +λ}

## Citation

If you use this code or findings, please cite:

```bibtex
@article{nsr2024,
  title={The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning},
  journal={arXiv preprint arXiv:2506.01347},
  year={2024},
  url={https://arxiv.org/pdf/2506.01347}
}
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size or GPU memory utilization:

```bash
python experiment.py \
    --objective NSR \
    --max_tokens 256 \
    --rollout_batch_size 64 \
    --gpu_mem_util 0.3
```

## License

MIT License. See original paper for academic usage guidelines.

## Acknowledgments

- Paper authors for the NSR/PSR/W-REINFORCE framework
- HuggingFace for Transformers and Datasets libraries
- vLLM team for efficient LLM inference
- Dataset: `justinphan3110/Countdown-Tasks-3to4`
