#!/usr/bin/env python3
"""Verification script to test the modular structure."""

import sys


def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")
    
    try:
        from nsr_countdown.config import TrainingConfig
        print("✓ config.py")
        
        from nsr_countdown.data import reward_fn, load_countdown_dataset, TEMPLATE
        print("✓ data.py")
        
        from nsr_countdown.model import init_policy, init_vllm, init_sampling_params
        print("✓ model.py")
        
        from nsr_countdown.tokenization import tokenize_prompt_and_output, get_response_log_probs
        print("✓ tokenization.py")
        
        from nsr_countdown.rl_objectives import RLObjective, make_weighted_rewards
        print("✓ rl_objectives.py")
        
        from nsr_countdown.rl_loss import (
            compute_group_normalized_advantages,
            compute_loss,
            masked_mean,
            masked_mean_drgrpo
        )
        print("✓ rl_loss.py")
        
        from nsr_countdown.trainer import train, rollout_with_vllm
        print("✓ trainer.py")
        
        from nsr_countdown.evaluator import evaluate_model, log_train, log_eval
        print("✓ evaluator.py")
        
        from nsr_countdown.utils import duplicate_data, get_constant_schedule_with_warmup
        print("✓ utils.py")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration instantiation."""
    print("\nTesting configuration...")
    
    try:
        from nsr_countdown.config import TrainingConfig
        from nsr_countdown.rl_objectives import RLObjective
        
        config = TrainingConfig()
        assert config.model_id == "Qwen/Qwen3-1.7B"
        assert config.objective == RLObjective.W_REINFORCE
        assert config.lambda_psr == 0.1
        assert config.use_std_normalization == True
        
        print("✓ Default configuration works")
        
        # Test property methods
        assert config.max_completion_length == config.max_tokens
        print("✓ Configuration properties work")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_reward_function():
    """Test the reward function with sample inputs."""
    print("\nTesting reward function...")
    
    try:
        from nsr_countdown.data import reward_fn
        
        # Test correct solution
        response = "Let me think... <answer>(1 + 2) * 3 - 4</answer>"
        ground_truth = {"target": 5, "numbers": [1, 2, 3, 4]}
        reward = reward_fn(response, ground_truth)
        assert reward == 1.0, f"Expected 1.0, got {reward}"
        print("✓ Correct solution gets +1.0 reward")
        
        # Test incorrect solution
        response = "Let me think... <answer>1 + 2 + 3</answer>"
        reward = reward_fn(response, ground_truth)
        assert reward == -1.0, f"Expected -1.0, got {reward}"
        print("✓ Incorrect solution gets -1.0 reward")
        
        # Test no answer tag
        response = "Let me think... 1 + 2 + 3 = 6"
        reward = reward_fn(response, ground_truth)
        assert reward == -1.0, f"Expected -1.0, got {reward}"
        print("✓ Missing answer tag gets -1.0 reward")
        
        return True
    except Exception as e:
        print(f"✗ Reward function test failed: {e}")
        return False


def test_rl_objectives():
    """Test RL objective implementations."""
    print("\nTesting RL objectives...")
    
    try:
        from nsr_countdown.rl_objectives import RLObjective, make_weighted_rewards
        from nsr_countdown.data import reward_fn
        
        # Mock data
        responses = ["correct", "incorrect", "correct", "incorrect"]
        ground_truths = [
            {"target": 5, "numbers": [1, 2, 3, 4]},
            {"target": 10, "numbers": [1, 2, 3, 4]},
            {"target": 5, "numbers": [1, 2, 3, 4]},
            {"target": 10, "numbers": [1, 2, 3, 4]},
        ]
        
        # Mock reward function for testing
        def mock_reward_fn(resp, gt):
            return 1.0 if "correct" in resp else -1.0
        
        # Test RLVR (all samples)
        rewards, keep_mask = make_weighted_rewards(
            responses, ground_truths, mock_reward_fn, RLObjective.RLVR
        )
        assert keep_mask.sum() == 4, "RLVR should keep all samples"
        print("✓ RLVR keeps all samples")
        
        # Test PSR (only correct)
        rewards, keep_mask = make_weighted_rewards(
            responses, ground_truths, mock_reward_fn, RLObjective.PSR
        )
        assert keep_mask.sum() == 2, "PSR should keep only correct samples"
        print("✓ PSR keeps only correct samples")
        
        # Test NSR (only incorrect)
        rewards, keep_mask = make_weighted_rewards(
            responses, ground_truths, mock_reward_fn, RLObjective.NSR
        )
        assert keep_mask.sum() == 2, "NSR should keep only incorrect samples"
        print("✓ NSR keeps only incorrect samples")
        
        # Test W-REINFORCE (all with different weights)
        rewards, keep_mask = make_weighted_rewards(
            responses, ground_truths, mock_reward_fn, RLObjective.W_REINFORCE, lambda_psr=0.1
        )
        assert keep_mask.sum() == 4, "W-REINFORCE should keep all samples"
        assert (rewards[0] == 0.1 and rewards[1] == -1.0), "W-REINFORCE should weight differently"
        print("✓ W-REINFORCE applies correct weights")
        
        return True
    except Exception as e:
        print(f"✗ RL objectives test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions."""
    print("\nTesting utilities...")
    
    try:
        from nsr_countdown.utils import duplicate_data
        
        arr = [1, 2, 3]
        result = duplicate_data(arr, 2)
        assert result == [1, 1, 2, 2, 3, 3], f"Expected [1,1,2,2,3,3], got {result}"
        print("✓ duplicate_data works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("NSR Implementation Verification")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_config()
    all_passed &= test_reward_function()
    all_passed &= test_rl_objectives()
    all_passed &= test_utils()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe modular structure is working correctly!")
        print("You can now run: python main.py")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

