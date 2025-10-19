#!/usr/bin/env python3
"""
Compare Python vs Rust reward evaluation on real Code-R1 data
"""
import sys
import time
from datasets import load_dataset

# Import Rust implementation
try:
    import fastrlrewards
except ImportError:
    print("ERROR: fastrlrewards not available. Build with: maturin develop --release")
    sys.exit(1)

# Import Python implementation
from train_grpo_coder1 import (
    format_reward as py_format_reward,
    execution_reward as py_execution_reward,
    CONFIG
)

def benchmark_comparison(num_samples=50):
    """
    Compare Python (parallel) vs Rust (sequential) on real dataset
    """
    print("\n" + "="*80)
    print("BENCHMARKING: Python vs Rust Reward Evaluation")
    print("="*80 + "\n")
    
    # Disable debug mode
    CONFIG.debug_mode = False
    
    # Load dataset
    print("Loading Code-R1 dataset...")
    dataset = load_dataset("ganler/code-r1-12k", split="train")
    
    # Filter: must have test, entry_point, AND completion
    dataset = dataset.filter(
        lambda x: x['test'] is not None 
        and x['test'] != 'null'
        and x['entry_point'] is not None
        and x['entry_point'] != 'null'
        and x['completion'] is not None  # ← KEY: Must have completion!
        and x['completion'] != 'null'
    )
    
    print(f"Filtered to {len(dataset)} samples with completions")
    test_dataset = dataset.shuffle(seed=42).select(range(num_samples))
    print(f"Selected {num_samples} samples for testing\n")
    
    # Use REAL completions from dataset
    completions = [sample['completion'] for sample in test_dataset]
    tests = [sample['test'] for sample in test_dataset]
    entry_points = [sample['entry_point'] for sample in test_dataset]
    
    # Show sample
    print("Sample completion (first 200 chars):")
    print(completions[0][:200] + "...\n")
    
    kwargs = {
        'test': tests,
        'entry_point': entry_points,
    }
    
    # Benchmark Python (with ProcessPoolExecutor)
    print("Running Python execution_reward (ProcessPoolExecutor)...")
    start = time.time()
    py_rewards = py_execution_reward(completions, **kwargs)
    py_time = time.time() - start
    print(f"Python completed in {py_time:.2f}s\n")
    
    # Benchmark Rust (single-threaded)
    print("Running Rust execution_reward (single-threaded)...")
    start = time.time()
    rust_rewards = fastrlrewards.execution_reward(completions, **kwargs)
    rust_time = time.time() - start
    print(f"Rust completed in {rust_time:.2f}s\n")
    
    # Compare results
    matches = sum(1 for p, r in zip(py_rewards, rust_rewards) if abs(p - r) < 0.01)
    py_pass = sum(py_rewards)
    rust_pass = sum(rust_rewards)
    
    # Results
    print("="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"Samples tested:              {num_samples}")
    print(f"Python time (parallel):      {py_time:.2f}s")
    print(f"Rust time (sequential):      {rust_time:.2f}s")
    
    if rust_time < py_time:
        print(f"Speedup:                     {py_time/rust_time:.2f}x FASTER ✓")
    else:
        print(f"Speedup:                     {py_time/rust_time:.2f}x (SLOWER - expected for sequential)")
    
    print(f"Results match:               {matches}/{num_samples} ({matches/num_samples*100:.1f}%)")
    print(f"Python: {int(py_pass)} tests passed")
    print(f"Rust:   {int(rust_pass)} tests passed")
    print(f"Avg time per completion:")
    print(f"  Python:                    {py_time/num_samples*1000:.1f}ms")
    print(f"  Rust:                      {rust_time/num_samples*1000:.1f}ms")
    print("="*80 + "\n")
    
    # Verify correctness
    if matches == num_samples:
        print("✓ CORRECTNESS: Perfect match - Rust implementation is correct!")
    else:
        print(f"✗ WARNING: {num_samples - matches} mismatches detected!")
        print("\nFirst few mismatches:")
        shown = 0
        for i, (p, r) in enumerate(zip(py_rewards, rust_rewards)):
            if abs(p - r) >= 0.01 and shown < 5:
                print(f"  Sample {i}: Python={p:.3f}, Rust={r:.3f}")
                shown += 1
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    if matches == num_samples:
        if rust_time > py_time:
            print("✓ Correctness validated")
            print("⚠ Sequential Rust is slower than parallel Python (expected)")
            print("→ Next step: Add Rayon parallelization to Rust")
            print(f"→ Target: <{py_time:.1f}s with parallel Rust")
        else:
            print("✓ Correctness validated")
            print("✓ Already faster! (unexpected but good)")
    else:
        print("✗ Fix correctness issues before proceeding")
    print("="*80 + "\n")

def test_format_reward(num_samples=20):
    """Quick test of format_reward"""
    print("\n" + "="*80)
    print("TESTING: format_reward")
    print("="*80 + "\n")
    
    completions = [
        "<think>reasoning</think>\n<answer>code</answer>",  # Valid
        "just code without format",  # Invalid
        "<think>x</think>\n<answer>y</answer>",  # Valid
    ] * (num_samples // 3)
    
    start = time.time()
    py_rewards = py_format_reward(completions)
    py_time = time.time() - start
    
    start = time.time()
    rust_rewards = fastrlrewards.format_reward(completions)
    rust_time = time.time() - start
    
    matches = sum(1 for p, r in zip(py_rewards, rust_rewards) if abs(p - r) < 0.01)
    
    print(f"Python time: {py_time*1000:.2f}ms")
    print(f"Rust time:   {rust_time*1000:.2f}ms")
    print(f"Speedup:     {py_time/rust_time:.2f}x")
    print(f"Match rate:  {matches}/{len(completions)}")
    
    if matches == len(completions):
        print("✓ format_reward: PASS\n")
    else:
        print("✗ format_reward: FAIL\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Python vs Rust rewards')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples to test')
    parser.add_argument('--format-only', action='store_true', help='Only test format_reward')
    
    args = parser.parse_args()
    
    if args.format_only:
        test_format_reward()
    else:
        # Test both
        test_format_reward()
        benchmark_comparison(args.samples)
