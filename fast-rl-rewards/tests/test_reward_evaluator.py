#!/usr/bin/env python3
"""
Tests for the complete reward evaluator
"""

import sys
import fastrlrewards

def test_format_reward_function():
    """Test module-level format_reward function"""
    completions = [
        "<think>Let me solve this.</think>\n<answer>def add(a, b): return a + b</answer>",
        "def add(a, b): return a + b",  # Missing format
        "<think>Thinking...</think>\n<answer>pass</answer>",
    ]
    
    rewards = fastrlrewards.format_reward(completions)
    assert len(rewards) == 3
    assert rewards[0] == 1.0  # Has format
    assert rewards[1] == 0.0  # No format
    assert rewards[2] == 1.0  # Has format
    print("✓ test_format_reward_function passed")

def test_execution_reward_function():
    """Test module-level execution_reward function"""
    completions = [
        "<answer>def add(a, b): return a + b</answer>",
        "<answer>def add(a, b): return a - b</answer>",  # Wrong
    ]
    
    # FIX: Proper Python formatting with newlines and indentation
    tests = [
        """def check(candidate):
    assert candidate(2, 3) == 5
    assert candidate(1, 1) == 2""",
        """def check(candidate):
    assert candidate(2, 3) == 5
    assert candidate(1, 1) == 2""",
    ]
    
    entry_points = ["add", "add"]
    
    rewards = fastrlrewards.execution_reward(
        completions,
        test=tests,
        entry_point=entry_points
    )
    
    assert len(rewards) == 2
    assert rewards[0] == 1.0  # Correct
    assert rewards[1] == 0.0  # Wrong
    print("✓ test_execution_reward_function passed")

def test_evaluator_class():
    """Test the evaluator class interface"""
    evaluator = fastrlrewards.RewardEvaluator(timeout_seconds=10)

    completions = [
        "<think>test</think>\n<answer>def foo(): return 42</answer>"
    ]
    tests = ["def check(candidate):\n    assert candidate() == 42"]
    entry_points = ["foo"]

    format_rewards = evaluator.format_reward(completions)
    exec_rewards = evaluator.execution_reward(
        completions, test=tests, entry_point=entry_points
    )

    assert format_rewards[0] == 1.0
    assert exec_rewards[0] == 1.0
    print("✓ test_evaluator_class passed")

def test_trl_dict_format():
    """Test TRL-style dict completion format"""
    completions = [
        {"content": "<think>ok</think>\n<answer>def square(x): return x * x</answer>"}
    ]
    
    kwargs = {
        "test": ["def check(candidate):\n    assert candidate(4) == 16"],
        "entry_point": ["square"],
    }
    
    rewards = fastrlrewards.execution_reward(completions, **kwargs)
    assert len(rewards) == 1
    assert rewards[0] == 1.0
    print("✓ test_trl_dict_format passed")

def test_multiple_evaluators():
    """Test that multiple evaluator instances work correctly"""
    eval1 = fastrlrewards.RewardEvaluator(timeout_seconds=5)
    eval2 = fastrlrewards.RewardEvaluator(timeout_seconds=20)
    
    completions = ["<think>x</think>\n<answer>pass</answer>"]
    
    r1 = eval1.format_reward(completions)
    r2 = eval2.format_reward(completions)
    
    assert r1 == r2 == [1.0]
    print("✓ test_multiple_evaluators passed")

if __name__ == "__main__":
    print("\nRunning reward evaluator tests...\n")
    test_format_reward_function()
    test_execution_reward_function()
    test_evaluator_class()
    test_trl_dict_format()
    test_multiple_evaluators()
    print("\n✅ All tests passed!\n")
