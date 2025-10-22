#!/bin/bash
# Launcher script for running all 6 NSR experiments
# Usage: bash run_experiments.sh

echo "========================================================================"
echo "Starting NSR Experiment Suite - 6 Experiments"
echo "========================================================================"
echo ""

# Experiment 1: NSR with 256 tokens
echo "Running Experiment 1/6: NSR with 256 tokens"
python experiment.py \
    --objective NSR \
    --max_tokens 256 \
    --seed 42

# Experiment 2: NSR with 512 tokens  
echo ""
echo "Running Experiment 2/6: NSR with 512 tokens"
python experiment.py \
    --objective NSR \
    --max_tokens 512 \
    --seed 42

# Experiment 3: PSR with 256 tokens
echo ""
echo "Running Experiment 3/6: PSR with 256 tokens"
python experiment.py \
    --objective PSR \
    --max_tokens 256 \
    --seed 42

# Experiment 4: PSR with 512 tokens
echo ""
echo "Running Experiment 4/6: PSR with 512 tokens"
python experiment.py \
    --objective PSR \
    --max_tokens 512 \
    --seed 42

# Experiment 5: W-REINFORCE with 256 tokens
echo ""
echo "Running Experiment 5/6: W-REINFORCE (lambda=0.3) with 256 tokens"
python experiment.py \
    --objective W_REINFORCE \
    --max_tokens 256 \
    --lambda_psr 0.3 \
    --seed 42

# Experiment 6: W-REINFORCE with 512 tokens
echo ""
echo "Running Experiment 6/6: W-REINFORCE (lambda=0.3) with 512 tokens"
python experiment.py \
    --objective W_REINFORCE \
    --max_tokens 512 \
    --lambda_psr 0.3 \
    --seed 42

# Experiment 7: PSR with 1024 tokens
echo ""
echo "Running Experiment 4/6: PSR with 512 tokens"
python experiment.py \
    --objective PSR \
    --max_tokens 512 \
    --seed 42

# Experiment 8: W-REINFORCE with 1024 tokens
echo ""
echo "Running Experiment 5/6: W-REINFORCE (lambda=0.3) with 256 tokens"
python experiment.py \
    --objective W_REINFORCE \
    --max_tokens 256 \
    --lambda_psr 0.3 \
    --seed 42

# Experiment 9: W-REINFORCE with 1024 tokens
echo ""
echo "Running Experiment 6/6: W-REINFORCE (lambda=0.3) with 512 tokens"
python experiment.py \
    --objective W_REINFORCE \
    --max_tokens 512 \
    --lambda_psr 0.3 \
    --seed 42

echo ""
echo "========================================================================"
echo "All 6 experiments completed!"
echo "========================================================================"
echo ""
echo "Results saved to ./output/"
echo "  - Models: ./output/models/"
echo "  - TensorBoard logs: ./output/tb/"
echo ""
echo "To view results:"
echo "  tensorboard --logdir ./output/tb"

