# NSR Implementation for Math Countdown

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2506.01347)

Implementation of **"The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning"** for the Countdown mathematical reasoning task.

## Overview

This repository implements several reinforcement learning objectives for training language models on reasoning tasks:

- **RLVR** (Reinforcement Learning with Verifiable Rewards): Standard RL baseline using all samples
- **PSR** (Positive Sample Reinforcement): Train only on correct samples
- **NSR** (Negative Sample Reinforcement): Train only on incorrect samples (the paper's key innovation)
- **W-REINFORCE** (Weighted REINFORCE): Weight correct samples with λ, incorrect with -1 (paper's best method, λ=0.1)

The implementation uses GRPO (Group Relative Policy Optimization) as the base algorithm and supports both GRPO and DR-GRPO loss variants.

## Repository Structure

```
.
├── nsr_countdown/              # Main package
│   ├── __init__.py
│   ├── config.py               # Centralized configuration (dataclass-based)
│   ├── data.py                 # Countdown task: prompts, rewards, validation
│   ├── model.py                # Model initialization (policy + vLLM)
│   ├── tokenization.py         # Tokenization utilities
│   ├── rl_objectives.py        # RL objectives (RLVR/PSR/NSR/W-REINFORCE)
│   ├── rl_loss.py              # Advantage computation and PPO loss
│   ├── trainer.py              # Training loop orchestration
│   ├── evaluator.py            # Evaluation and logging
│   └── utils.py                # Miscellaneous helpers
│
├── main.py                     # Entry point for training
├── requirements.txt            # Python dependencies
├── rlvr_decomposed_math_countdown.py  # Original monolithic implementation
└── README.md                   # This file
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Centralized hyperparameters using dataclass |
| `data.py` | Task-specific logic (reward function, validation, dataset loading) |
| `model.py` | Policy model and vLLM engine initialization |
| `tokenization.py` | Tokenization and log probability computation |
| `rl_objectives.py` | RL objective implementations (NSR/PSR/W-REINFORCE) |
| `rl_loss.py` | Advantage computation, PPO clipped loss, masked averaging |
| `trainer.py` | Main training loop with rollout generation and optimization |
| `evaluator.py` | Evaluation metrics and TensorBoard logging |
| `utils.py` | Learning rate scheduling and utility functions |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd NPR-Implementation-MathCountdown-1

# Install dependencies
pip install -r requirements.txt

# Optional: Install flash-attention for faster training
pip install flash-attn --no-build-isolation
```

**Requirements:**
- Python 3.10+
- CUDA-capable GPU (recommended: 24GB+ VRAM for Qwen 1.7B)
- PyTorch 2.0+

## Quick Start

### Basic Training

```bash
python main.py
```

This will train using the default configuration:
- Model: `Qwen/Qwen3-1.7B`
- Objective: `W-REINFORCE` (λ=0.1)
- Loss: `GRPO`
- Steps: 200

### Customizing Configuration

Modify `main.py` to change training settings:

```python
from nsr_countdown.config import TrainingConfig
from nsr_countdown.rl_objectives import RLObjective

config = TrainingConfig()

# Change RL objective
config.objective = RLObjective.NSR  # Train only on mistakes

# Adjust hyperparameters
config.n_grpo_steps = 500
config.lambda_psr = 0.2  # For W-REINFORCE
config.loss_type = "dr_grpo"  # Use DR-GRPO variant

# Model settings
config.model_id = "Qwen/Qwen3-1.7B"
config.rollout_batch_size = 128
config.group_size = 8
```

### RL Objectives

Select the training objective by setting `config.objective`:

```python
from nsr_countdown.rl_objectives import RLObjective

# Standard RL baseline
config.objective = RLObjective.RLVR

# Positive Sample Reinforcement (only correct samples)
config.objective = RLObjective.PSR

# Negative Sample Reinforcement (only incorrect samples)
config.objective = RLObjective.NSR

# Weighted REINFORCE (paper's best, λ=0.1 recommended)
config.objective = RLObjective.W_REINFORCE
config.lambda_psr = 0.1
```

## Key Implementation Details

### Binary Reward Function

The Countdown task uses binary verifiable rewards:
- **+1**: Equation is valid, uses correct numbers, equals target
- **-1**: Otherwise (invalid equation, wrong numbers, or incorrect result)

See `data.py:reward_fn()` for implementation.

### Group-Normalized Advantages

GRPO computes advantages by grouping multiple samples per prompt and normalizing within groups:

```python
advantages = (reward - group_mean) / (group_std + ε)
```

This reduces variance and improves training stability. See `rl_loss.py:compute_group_normalized_advantages()`.

### NSR/PSR Filtering

The `make_weighted_rewards()` function (in `rl_objectives.py`) applies objective-specific sample filtering:

- **NSR**: Keeps only incorrect samples (`reward < 0`)
- **PSR**: Keeps only correct samples (`reward > 0`)
- **W-REINFORCE**: Keeps all samples but reweights (+λ for correct, -1 for incorrect)

### Loss Types

Two loss variants are supported:

1. **GRPO** (default): Mean over response tokens per sequence, then average over batch
2. **DR-GRPO**: Normalize by fixed `max_tokens` instead of actual token count

Set via `config.loss_type = "grpo"` or `"dr_grpo"`.

## Monitoring Training

Training logs are saved to TensorBoard:

```bash
tensorboard --logdir=./output/tb
```

### Key Metrics

**Training:**
- `train/loss`: Policy gradient loss
- `train/reward_mean`: Average reward over rollout batch
- `samples/correct_ratio`: Fraction of correct solutions (learning progress)
- `filtering/samples_kept`: Number of samples kept after NSR/PSR filtering

**Evaluation:**
- `eval/accuracy`: Percentage of correct solutions
- `eval/mean_reward`: Average reward on evaluation set
- `eval/examples_*`: Sample solutions (correct/partial/failed)

## Countdown Task

The Countdown task challenges the model to create equations using given numbers to reach a target:

**Example:**
- Numbers: `[1, 2, 3, 4]`
- Target: `5`
- Solution: `(1 + 2) * 3 - 4 = 5`

**Dataset:** `justinphan3110/Countdown-Tasks-3to4` (numbers with 3-4 digits)

## Extending to Other Tasks

To adapt this implementation to other reasoning tasks:

1. **Modify `data.py`:**
   - Replace `reward_fn()` with task-specific reward
   - Update `load_countdown_dataset()` for your dataset
   - Adjust prompt template

2. **Keep `rl_objectives.py`, `rl_loss.py`, `trainer.py` unchanged** (task-agnostic)

3. **Update `config.py`** with task-specific hyperparameters

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{nsr2025,
  title={The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning},
  author={[Authors]},
  journal={arXiv preprint arXiv:2506.01347},
  year={2025}
}
```

## License

MIT License - see original paper for academic usage guidelines.

## Troubleshooting

### Out of Memory

Reduce batch size or use gradient checkpointing:
```python
config.rollout_batch_size = 64
config.gradient_accumulation_steps = 16
config.gpu_memory_utilization = 0.3
```

### vLLM Issues

If vLLM fails to initialize, check:
- CUDA version compatibility
- GPU memory availability
- Try setting `os.environ["VLLM_USE_V1"] = "0"` (already done in `utils.py`)

### Import Errors

Ensure you're running from the repository root and the package is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python main.py
```

## Development

### Original Implementation

The original monolithic implementation is preserved in `rlvr_decomposed_math_countdown.py` for reference.

### Code Style

- Modules follow single responsibility principle
- Type hints used throughout
- Docstrings follow Google style

### Testing

To verify the installation works:
```bash
python -c "from nsr_countdown.config import TrainingConfig; print('✓ Installation successful')"
```

## Acknowledgments

- Original paper authors
- vLLM team for efficient inference
- HuggingFace for transformers and datasets
