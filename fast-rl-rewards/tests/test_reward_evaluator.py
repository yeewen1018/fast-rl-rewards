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

if __name__ == "__main__":
    print("\nRunning reward evaluator tests...\n")
    test_format_reward_function()
    print("\n✅ All tests passed!\n")