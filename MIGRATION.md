# Migration Guide: Monolithic → Modular

This guide explains how the original `rlvr_decomposed_math_countdown.py` (739 lines) was refactored into the modular structure.

## Comparison Table

| Original Location | New Location | Lines | Notes |
|-------------------|--------------|-------|-------|
| Lines 1-48 | README.md header | - | Documentation |
| Lines 50-68 | `utils.py` | 10 | Imports consolidated |
| Lines 70-73 | `utils.setup_logging()` | 4 | Logging setup |
| Lines 77-80 | `utils.get_constant_schedule_with_warmup()` | 15 | LR scheduler |
| Lines 85-88 | `data.TEMPLATE` | 4 | Prompt template |
| Lines 92-126 | `model.py` | 85 | vLLM + sampling setup |
| Lines 129-151 | `tokenization.py` | 55 | Tokenization utils |
| Lines 153-247 | `data.py` | 90 | Reward + validation |
| Lines 250-327 | `evaluator.py` | 160 | Evaluation + logging |
| Lines 329-362 | `rl_loss.compute_group_normalized_advantages()` | 40 | Advantage computation |
| Lines 364-368 | `rl_objectives.RLObjective` | 10 | Enum definition |
| Lines 370-404 | `rl_objectives.make_weighted_rewards()` | 50 | Objective logic |
| Lines 410-456 | `rl_loss.compute_loss()` | 40 | PPO loss |
| Lines 459-494 | `rl_loss.masked_mean*()` | 35 | Loss averaging |
| Lines 496-500 | `tokenization.get_response_log_probs()` | 10 | Log probs |
| Lines 504-518 | `trainer.rollout_with_vllm()` | 25 | Rollout generation |
| Lines 521-522 | (Inline in trainer) | 2 | Tokenize wrapper |
| Lines 525-542 | `trainer.grpo_microbatch_step()` | 35 | Microbatch step |
| Lines 545-653 | `trainer.train()` | 180 | Main training loop |
| Lines 656-662 | `model.init_policy()` | 12 | Policy init |
| Lines 665-736 | `main.py` | 100 | Entry point |
| - | `config.py` | 55 | **NEW** Centralized config |
| - | `examples/` | 200 | **NEW** Example scripts |
| - | README.md | 200 | **NEW** Comprehensive docs |

## Key Changes

### 1. Configuration Extraction

**Before:**
```python
# Scattered throughout main()
model_id = "Qwen/Qwen3-1.7B"
n_grpo_steps = 200
rollout_batch_size = 128
objective = RLObjective.W_REINFORCE
lambda_psr = 0.1
# ... 20+ more parameters
```

**After:**
```python
# config.py
@dataclass
class TrainingConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    n_grpo_steps: int = 200
    rollout_batch_size: int = 128
    objective: RLObjective = RLObjective.W_REINFORCE
    lambda_psr: float = 0.1
    # All in one place with type hints
```

### 2. Task-Specific Separation

**Before:** Countdown logic mixed with RL code

**After:** Clean separation in `data.py`
```python
# data.py (self-contained)
def reward_fn(generated_text, ground_truth):
    # Countdown-specific reward logic
    ...

def load_countdown_dataset(split, tokenizer, max_tokens):
    # Countdown-specific data loading
    ...
```

### 3. RL Components Modularized

**Before:** All RL logic in one file

**After:** Split by responsibility
- `rl_objectives.py`: Objective definitions (RLVR/PSR/NSR/W-REINFORCE)
- `rl_loss.py`: Advantage computation, PPO loss, masked averaging
- `trainer.py`: Training loop orchestration

### 4. Dependency Injection

**Before:** Global `reward_fn` used everywhere

**After:** Passed as parameter
```python
# trainer.py
def train(..., reward_fn: Callable, ...):
    # Uses injected reward_fn
    rewards = [reward_fn(r, gt) for r, gt in ...]
```

### 5. Import Structure

**Before:**
```python
# Single file, no imports needed
```

**After:**
```python
# main.py
from nsr_countdown.config import TrainingConfig
from nsr_countdown.data import load_countdown_dataset, reward_fn
from nsr_countdown.model import init_policy, init_vllm
from nsr_countdown.trainer import train
# Clear, explicit dependencies
```

## Line-by-Line Mapping

### Example: Reward Function

**Original** (`rlvr_decomposed_math_countdown.py:226-246`):
```python
def reward_fn(generated_text: str, ground_truth: Dict, scale_factor: float = 1.0) -> float:
    """Binary verifiable reward for Countdown..."""
    target = ground_truth.get("target")
    numbers = ground_truth.get("numbers", []) or ground_truth.get("nums", [])
    eq = _extract_answer(generated_text)
    if eq is None:
        return -1.0
    if not _validate_numbers(eq, numbers):
        return -1.0
    val = _evaluate_equation(eq)
    if val is None:
        return -1.0
    return 1.0 if abs(val - target) < 1e-6 else -1.0
```

**New** (`nsr_countdown/data.py:80-100`):
```python
def reward_fn(generated_text: str, ground_truth: Dict, scale_factor: float = 1.0) -> float:
    """
    Binary verifiable reward for Countdown:
    +1.0 if equation valid + uses exactly the given numbers + equals target,
    -1.0 otherwise.
    
    Args:
        generated_text: The model-generated response.
        ground_truth: Dictionary containing 'target' and 'numbers'/'nums'.
        scale_factor: Unused, kept for compatibility.
    
    Returns:
        +1.0 for correct solutions, -1.0 otherwise.
    """
    target = ground_truth.get("target")
    numbers = ground_truth.get("numbers", []) or ground_truth.get("nums", [])
    eq = _extract_answer(generated_text)
    if eq is None:
        return -1.0
    if not _validate_numbers(eq, numbers):
        return -1.0
    val = _evaluate_equation(eq)
    if val is None:
        return -1.0
    return 1.0 if abs(val - target) < 1e-6 else -1.0
```

**Changes:** Added comprehensive docstring, otherwise identical logic.

### Example: Training Loop

**Original** (`rlvr_decomposed_math_countdown.py:545-653`):
- 108 lines in single function
- Mixed: sampling, rollout, filtering, advantages, microbatching, logging

**New** (`nsr_countdown/trainer.py:118-280`):
- Same logic, but:
  - Helper function `rollout_with_vllm()` extracted
  - Helper function `grpo_microbatch_step()` extracted
  - Clear subsection comments
  - Evaluation logic moved to `evaluator.py`

## Running the Original vs Modular

### Original

```bash
# Edit lines 666-667 to change objective
python rlvr_decomposed_math_countdown.py
```

### Modular

```bash
# Edit config in main.py or create custom script
python main.py

# Or use examples
python examples/custom_training.py
```

## What Was Added (Not in Original)

1. **`config.py`**: Centralized configuration (55 lines)
2. **`examples/`**: Example scripts showing usage (200+ lines)
3. **README.md**: Comprehensive documentation (250+ lines)
4. **STRUCTURE.md**: Architecture documentation (this file)
5. **verify_installation.py**: Testing script (150 lines)
6. **`.gitignore`**: Git configuration
7. **Docstrings**: Every function documented with Args/Returns
8. **Type hints**: Complete type annotations throughout

## What Stayed the Same

- **All algorithms**: PPO loss, GRPO, advantages, objectives
- **All hyperparameters**: Default values unchanged
- **All task logic**: Reward function, validation, dataset
- **All evaluation**: Metrics, logging, TensorBoard

## Benefits of Refactoring

| Aspect | Original | Modular |
|--------|----------|---------|
| **Testability** | Hard to test components | Each module testable |
| **Reusability** | Copy-paste | Import what you need |
| **Readability** | 739-line file | Max 240 lines per file |
| **Extensibility** | Modify main file | Add new modules |
| **Documentation** | Minimal | Comprehensive |
| **Collaboration** | One file, conflicts | Multiple files, parallel work |
| **Debugging** | Stack traces unclear | Clear module boundaries |

## Migration Checklist

If you want to migrate custom modifications:

- [ ] Identify which lines you changed in the original
- [ ] Find corresponding location in modular structure (see table above)
- [ ] Apply your changes to the new file
- [ ] Update imports if needed
- [ ] Run `python verify_installation.py` to test
- [ ] Run your training to ensure it works

## Example: Adding Custom Reward

**Original approach:**
```python
# Edit reward_fn() in rlvr_decomposed_math_countdown.py line 226
```

**Modular approach:**
```python
# 1. Edit nsr_countdown/data.py:reward_fn()
def reward_fn(generated_text, ground_truth):
    # Your custom logic
    ...

# 2. main.py automatically uses it via import
# No other changes needed!
```

## Example: Changing Hyperparameters

**Original approach:**
```python
# Edit main() line 669-679
n_grpo_steps = 500  # Change this
rollout_batch_size = 256  # And this
# Scattered across 50+ lines
```

**Modular approach:**
```python
# Edit config or main.py
config = TrainingConfig()
config.n_grpo_steps = 500
config.rollout_batch_size = 256
# All in one place
```

## Backward Compatibility

The original file is preserved as `rlvr_decomposed_math_countdown.py` for reference and compatibility. You can:

1. Keep using the original if it works for you
2. Gradually migrate to modular structure
3. Use both during transition period

## Questions?

See `STRUCTURE.md` for architecture details or `examples/` for usage patterns.

