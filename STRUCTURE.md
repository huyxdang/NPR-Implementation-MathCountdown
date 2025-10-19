# Code Structure Documentation

This document explains the modular structure of the NSR implementation and how components interact.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                 │
│                    (Entry point & orchestration)                 │
└────────┬──────────────────────────────────────────────┬─────────┘
         │                                                │
         ├── config.py                                   │
         │   (Centralized hyperparameters)               │
         │                                                │
         ├── data.py ◄────────────────────────────────┐  │
         │   (Task-specific: rewards, validation)     │  │
         │                                             │  │
         ├── model.py                                  │  │
         │   (Policy + vLLM initialization)            │  │
         │                                             │  │
         └── trainer.py ◄─────────────────────────────┼──┘
             │                                         │
             ├── rl_objectives.py ◄─────────────────┐ │
             │   (NSR/PSR/W-REINFORCE logic)         │ │
             │                                        │ │
             ├── rl_loss.py                          │ │
             │   (Advantages, PPO loss)              │ │
             │                                        │ │
             ├── tokenization.py                     │ │
             │   (Tokenization utilities)            │ │
             │                                        │ │
             ├── evaluator.py ──────────────────────►│─┘
             │   (Evaluation & logging)              │
             │                                        │
             └── utils.py                            │
                 (Scheduler, helpers)                │
                                                      │
                 Circular dependency resolved: ◄─────┘
                 evaluator uses data.reward_fn via import
```

## Module Dependencies

### Core Modules (No dependencies on other modules)

1. **`config.py`**
   - Only depends on `rl_objectives.RLObjective` for type hints
   - Defines `TrainingConfig` dataclass
   - No external package dependencies

2. **`data.py`**
   - Self-contained task logic
   - Dependencies: `datasets`, `transformers.AutoTokenizer`
   - Exports: `reward_fn()`, `load_countdown_dataset()`, validation functions

3. **`rl_objectives.py`**
   - Self-contained RL objective implementations
   - Dependencies: `torch`, standard library
   - Exports: `RLObjective` enum, `make_weighted_rewards()`

### Mid-level Modules

4. **`model.py`**
   - Model initialization utilities
   - Dependencies: `transformers`, `vllm`, `torch`
   - No internal module dependencies

5. **`tokenization.py`**
   - Tokenization helpers
   - Dependencies: `torch`, `transformers`
   - No internal module dependencies

6. **`utils.py`**
   - Generic utilities
   - Dependencies: `torch`
   - No internal module dependencies

7. **`rl_loss.py`**
   - RL algorithm components
   - Dependencies: `torch`
   - No internal module dependencies (reward_fn passed as parameter)

### High-level Modules

8. **`evaluator.py`**
   - Evaluation and logging
   - Internal dependencies: `data` (for `_extract_answer`, `_evaluate_equation`)
   - External dependencies: `torch`, `vllm`, `tensorboard`

9. **`trainer.py`**
   - Training orchestration
   - Internal dependencies:
     - `model` (vLLM loading)
     - `tokenization` (tokenize rollouts)
     - `rl_objectives` (reward weighting)
     - `rl_loss` (advantages, PPO loss)
     - `utils` (duplicate_data)
     - `evaluator` (evaluation, logging)
   - External dependencies: `torch`, `transformers`, `vllm`

## Data Flow

### Training Step

```
1. main.py
   ├─ Loads TrainingConfig
   ├─ Initializes policy (model.init_policy)
   ├─ Initializes vLLM (model.init_vllm)
   ├─ Loads dataset (data.load_countdown_dataset)
   └─ Calls trainer.train()

2. trainer.train() [Single step]
   ├─ Samples prompts from training set
   ├─ Generates rollouts with vLLM
   ├─ Computes rewards (data.reward_fn)
   ├─ Applies objective-specific weighting (rl_objectives.make_weighted_rewards)
   ├─ Filters samples if PSR/NSR
   ├─ Computes advantages (rl_loss.compute_group_normalized_advantages)
   ├─ Tokenizes rollouts (tokenization.tokenize_prompt_and_output)
   ├─ For each microbatch:
   │  ├─ Computes log probs (tokenization.get_response_log_probs)
   │  ├─ Computes PPO loss (rl_loss.compute_loss)
   │  └─ Averages loss (rl_loss.masked_mean)
   ├─ Updates policy parameters
   └─ Periodically evaluates (evaluator.evaluate_model)
```

### Evaluation Flow

```
evaluator.evaluate_model()
├─ Generates completions with vLLM
├─ For each completion:
│  ├─ Computes reward (data.reward_fn)
│  ├─ Extracts equation (data._extract_answer)
│  └─ Evaluates equation (data._evaluate_equation)
├─ Aggregates metrics
└─ Returns metrics dictionary
```

## Key Design Patterns

### 1. Dependency Injection

Instead of hardcoding dependencies, functions accept them as parameters:

```python
# trainer.py
def train(
    ...,
    reward_fn: Callable,  # Injected from data.py
    ...
):
    # Uses reward_fn without importing data.py
```

This keeps modules decoupled and testable.

### 2. Configuration as Data

All hyperparameters in one place:

```python
# config.py
@dataclass
class TrainingConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    n_grpo_steps: int = 200
    ...
```

Makes experiments reproducible and easy to document.

### 3. Single Responsibility Principle

Each module has one clear purpose:
- `data.py`: Task-specific logic only
- `rl_objectives.py`: RL objective implementations only
- `trainer.py`: Training loop orchestration only

### 4. Enum-based Strategy Pattern

Different RL objectives via enum:

```python
# rl_objectives.py
class RLObjective(str, Enum):
    NSR = "nsr"
    PSR = "psr"
    W_REINFORCE = "w_reinforce"
    RLVR = "rlvr"

def make_weighted_rewards(..., objective: RLObjective):
    if objective == RLObjective.NSR:
        # NSR logic
    elif objective == RLObjective.PSR:
        # PSR logic
    ...
```

Simple, explicit, easy to extend.

## Extension Points

### Adding a New RL Objective

1. Add enum value to `rl_objectives.RLObjective`
2. Add logic in `make_weighted_rewards()`
3. Update `config.py` default if needed

### Adding a New Task

1. Create new `data_<task>.py` with:
   - Task-specific `reward_fn()`
   - Task-specific dataset loader
   - Task-specific validation
2. Update `main.py` to import from new data module
3. Keep all other modules unchanged

### Adding a New Loss Type

1. Add function to `rl_loss.py` (e.g., `masked_mean_custom()`)
2. Update `trainer.grpo_microbatch_step()` to handle new loss type
3. Add option to `config.loss_type`

## Testing Strategy

Each module can be tested independently:

```python
# Test data module
from nsr_countdown.data import reward_fn
assert reward_fn("...answer...", {"target": 5, "numbers": [...]}) == 1.0

# Test RL objectives
from nsr_countdown.rl_objectives import make_weighted_rewards, RLObjective
rewards, mask = make_weighted_rewards(..., objective=RLObjective.NSR)
assert mask.sum() == num_incorrect_samples

# Test utilities
from nsr_countdown.utils import duplicate_data
assert duplicate_data([1, 2], 3) == [1, 1, 1, 2, 2, 2]
```

See `verify_installation.py` for complete test examples.

## File Sizes

Approximate line counts for each module:

| Module | Lines | Complexity |
|--------|-------|------------|
| `config.py` | ~60 | Low |
| `data.py` | ~140 | Medium |
| `model.py` | ~90 | Low |
| `tokenization.py` | ~70 | Low |
| `rl_objectives.py` | ~100 | Medium |
| `rl_loss.py` | ~140 | Medium-High |
| `trainer.py` | ~240 | High |
| `evaluator.py` | ~160 | Medium |
| `utils.py` | ~50 | Low |
| **Total** | **~1050** | Modular |

Compare to original monolithic: ~740 lines (but less documented/structured).

## Benefits of This Structure

1. **Maintainability**: Each module has clear boundaries
2. **Testability**: Components can be tested in isolation
3. **Reusability**: RL components can be reused for other tasks
4. **Readability**: Clear separation of concerns
5. **Extensibility**: Easy to add new objectives, tasks, or loss functions
6. **Documentation**: Each module is self-documenting with docstrings
7. **Collaboration**: Multiple people can work on different modules
8. **Research-standard**: Follows best practices for ML research code

