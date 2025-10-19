# Example Scripts

This directory contains example scripts demonstrating how to use the modular NSR implementation.

## Scripts

### `custom_training.py`

Demonstrates how to run different training experiments with various RL objectives:

```bash
python examples/custom_training.py
```

**Experiments included:**
- `experiment_nsr_only()`: Train only on incorrect samples (NSR)
- `experiment_psr_only()`: Train only on correct samples (PSR)
- `experiment_w_reinforce()`: Weighted REINFORCE (paper's best, λ=0.1)
- `experiment_dr_grpo()`: DR-GRPO loss variant

Uncomment the desired experiment in the `__main__` section.

### `evaluate_model.py`

Shows how to evaluate a trained model checkpoint:

```bash
python examples/evaluate_model.py
```

Update the `checkpoint_path` variable to point to your trained model or use a base model to see baseline performance.

## Quick Start

1. **Train with W-REINFORCE (recommended):**
   ```python
   from examples.custom_training import experiment_w_reinforce
   experiment_w_reinforce()
   ```

2. **Evaluate a trained model:**
   ```python
   from examples.evaluate_model import evaluate_checkpoint
   evaluate_checkpoint("./output/nsr_model_<timestamp>")
   ```

3. **Compare NSR vs PSR:**
   Run both experiments and compare TensorBoard logs:
   ```bash
   tensorboard --logdir=./output/tb
   ```

## Customizing Experiments

All experiments use `TrainingConfig`. To customize:

```python
from nsr_countdown.config import TrainingConfig
from nsr_countdown.rl_objectives import RLObjective

config = TrainingConfig()
config.n_grpo_steps = 500           # More training steps
config.rollout_batch_size = 256     # Larger batch
config.objective = RLObjective.NSR  # Use NSR
config.lambda_psr = 0.2             # Different lambda for W-REINFORCE
config.learning_rate = 5e-7         # Lower learning rate

# Then pass config to run_experiment()
```

## Monitoring

All experiments log to TensorBoard:

```bash
tensorboard --logdir=./output/tb
```

Key metrics to watch:
- `eval/accuracy`: Solution accuracy (%)
- `samples/correct_ratio`: Learning progress
- `filtering/samples_kept`: Sample efficiency for NSR/PSR
- `train/reward_mean`: Training reward signal

