# 🦀 fast-rl-rewards

**High-performance Rust-based reward evaluator for RL training (PPO, GRPO, custom)**

## Overview
A scalable, high-performance reward evaluation framework for reinforcement learning. Built in Rust to offload CPU-heavy reward tasks - such as code execution, mathematical verification, tool-use scoring, and environment simulation - from Python RL workflows. This maximizes GPU utilization and enables faster, larger rollouts. 

<div align="center">
  <br>
  <figure>
  <img src="./assets/figure1.png" alt="fast-rl-rewards diagram" width="500%">
    <figcaption><b>Figure 1:</b> Reward evaluation can take as long as generation in Python RL workflows, creating a CPU bottleneck and GPU idle time. <code>fast-rl-rewards</code> addresses this bottleneck with Rust-based parallelization, reducing reward latency and improving GPU utilization. </figcaption>
  </figure>
  <br><br>
</div>

**Key Features**: 
- 🧵 **Native Thread-level Parallelism** - Leverages [**Rayon**](https://crates.io/crates/rayon), achieving **near-linear throughput scaling** until full CPU core saturation.
- 🔒 **Sandboxed Execution** - Provides robust, timeout-protected sandboxing (via [**firejail**](https://github.com/netblue30/firejail)) for secure reward evaluation.
- 🐍 **Python API via PyO3** — [**PyO3 bindings**](https://crates.io/crates/pyo3) enable seamless integration with Python RL frameworks (e.g. HuggingFace TRL, VERL)
- 🧩 **Built-in Reward Functions** - Exposes a suite of built-in Rust reward functions optimized for CPU-intensive tasks. 
- 🧠 **User-Defined Rewards (UDFs)** *(coming soon)* - Allows users to define custom reward logic in Python, while `fast-rl-rewards` automatically parallelizes and executes it efficiently in Rust.
- 🌐 **Distributed Evaluation** *(planned)* — Integrates with Ray for distributed reward computation across multiple nodes or GPUs.

## Get Started
#### **1. Installation**

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Create and activate a Python virtual environment
python -m venv rl-venv
source rl-venv/bin/activate

# Build and install fast-rl-rewards
cd fast-rl-rewards
maturin develop --release
```

#### **2. Example Usage (with HuggingFace TRL)**


```python
# Import Rust-backed reward functions
from fast_rl_rewards import format_reward, execution_reward

# Optional: use the RewardEvaluator class for advanced control
# from fast_rl_rewards import RewardEvaluator
# evaluator = RewardEvaluator(timeout_seconds=15, max_workers=32)
# reward_fns = [evaluator.format_reward, evaluator.execution_reward]

# Configure GRPO or PPO Trainer
training_args = GRPOConfig(
    # ... other parameters ...
    reward_functions=[
        format_reward,       # Rust-based format reward
        execution_reward     # Rust-based execution reward
    ],
)

trainer = GRPOTrainer(
    model=MODEL_NAME,
    args=training_args,
    train_dataset=dataset,
    # ... other setup ...
)

# Start training
trainer.train()

```
💡 See the examples/ directory for full integration examples.


## Performance and Scalability
<p align="center">
  <img src="./assets/profiling_results.png" alt="fast-rl-rewards diagram" width="500%">
  <fi
</p>

We profiled `fast-rl-rewards` on **Qwen2.5-Coder-7B-Instruct** fine-tuned with the **code-r1-12k** dataset under the following representative setup:

- 6 GPUs for text generation, 2 GPUs for reward evaluation  
- `num_generations = 32`, `per_device_batch_size = 8`, `gradient_accumulation_steps = 4`

This configuration reflects a **generation-optimal setting**, where inference throughput is saturated — highlighting reward computation as the primary bottleneck.


| Config | Reward Engine | Avg GPU Utilization | Avg CPU Exec Time (µs) | Step Time (µs) |
|:--------|:----------------|:-------------------:|:----------------------:|:---------------:|
| Python baseline | Python | 56.9% | 835,019 | 31,208,489 |
| fast-rl-rewards | Rust | **78.0%** | **554,420** | **23,030,929** |

> Profiling captured with PyTorch Profiler on 8×A100 (80 GB) GPUs.  
> Rust-based reward evaluation eliminates CPU bottlenecks, improving GPU utilization by **37%** and reducing step latency by **26%**.

---

## Scaling Results

To validate scaling efficiency, we benchmarked reward evaluation throughput across increasing rollout batch sizes.  
`fast-rl-rewards` maintains near-linear scaling up to full CPU core saturation, demonstrating efficient parallelization via Rayon.

| Config | # Completions | Python Reward (s) | Rust Reward (s) | Reward Speedup |
|:--------|:--------------:|:----------------:|:---------------:|:----------------:|
| Small | 3,072 | 11.21 | 2.26 | **4.96×** |
| Medium | 6,144 | 13.22 | 4.17 | **3.17×** |
| Large | 12,288 | 17.31 | 9.35 | **1.85×** |
| XL | 24,576 | 27.19 | 18.26 | **1.49×** |

> Benchmarked on 8×A100 (80 GB) GPUs.  
> `fast-rl-rewards` achieves up to **4.9× faster** reward computation at smaller rollout sizes, maintaining consistent scaling as batch size increases.

