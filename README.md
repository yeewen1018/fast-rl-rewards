# 🦀 fast-rl-rewards

**High-performance Rust-based reward evaluator for RL training (PPO, GRPO, custom)**

> Designed to remove CPU bottlenecks in reward computation tasks (e.g., code execution, logic/maths verification, and tool-use scoring). Scales linearly with rollouts and improves GPU utilization.

## Overview
A scalable, high-performance reward evaluation framework for reinforcement learning. Built in Rust to offload CPU-heavy reward tasks - such as code execution, mathematical verification, and environment simulation - from Python RL workflows. This maximizes GPU utilization and enables faster, larger rollouts. 

<p align="center">
    <img src="assets/figure1.jpg" alt="fast-rl-rewards diagram" width="600">
</p>
