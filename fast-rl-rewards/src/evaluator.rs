//! src/evaluator.rs
//!
//! Core reward evaluation logic.

use crate::extraction::extract_code_from_completion;
use crate::sandbox::run_sandboxed_tests;
use crate::test_wrapper::wrap_tests_for_complete_execution;
use anyhow::{Result, ensure};
use once_cell::sync::Lazy;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use regex::Regex;

// ==========================================================================================

/// Configuration for `RewardEvaluator`.
#[derive(Clone, Debug)]
pub struct EvaluatorConfig {
    /// Maximum wall-clock execution time per test in seconds.
    ///
    /// This is the total real-world time including CPU, I/O, and sleep.
    /// The process is killed if it exceeds this time regardless of CPU usage.
    pub timeout_seconds: u64,

    /// Memory limit for sandboxed execution in megabytes.
    ///
    /// Enforced by Firejail's `--rlimit-as` (address space limit).
    pub memory_limit_mb: u64,

    /// Maximum CPU time (user + system) per test in seconds.
    ///
    /// This counts only actual CPU usage. Enforced by Firejail's `--rlimit-cpu`.
    /// Should typically be set lower than `timeout_seconds`.
    pub cpu_time_limit: u64,

    /// Number of Rayon threads for parallel evaluation.
    ///
    /// - `Some(n)`: Use exactly `n` threads
    /// - `None`: Use default (number of CPU cores)
    pub num_threads: Option<usize>,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: 15,
            memory_limit_mb: 512,
            cpu_time_limit: 12,
            num_threads: Some(32),
        }
    }
}

impl EvaluatorConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.timeout_seconds > 0,
            "timeout_seconds (wall-clock timeout) must be at least 1, got {}",
            self.timeout_seconds
        );
        ensure!(
            self.memory_limit_mb >= 64,
            "memory_limit_mb must be at least 64MB for Python execution, got {}MB",
            self.memory_limit_mb
        );
        ensure!(
            self.cpu_time_limit > 0,
            "cpu_time_limit (CPU time limit) must be at least 1 second, got {}",
            self.cpu_time_limit
        );

        // Warn if timeout is lower than CPU limit (unusual but not invalid)
        if self.timeout_seconds < self.cpu_time_limit {
            eprintln!(
                "Warning: timeout_seconds ({}) is lower than cpu_time_limit ({}). \
                 Wall-clock timeout will likely be hit first.",
                self.timeout_seconds, self.cpu_time_limit
            );
        }

        Ok(())
    }
}

// ==========================================================================================

/// Main reward evaluator.
///
/// Orchestrates the reward evaluation workflow: code extraction from LLM outputs,
/// test code wrapping to run all assertions (preventing reward hacking), sandboxed
/// execution, and result aggregation. Uses Rayon to parallelize evaluation across batches.
///
/// # Examples
/// ```python
/// from fastrlrewards import RewardEvaluator
///
/// evaluator = RewardEvaluator(num_threads = 64, timeout_seconds = 20)
/// scores = evaluator.execution_reward(completions, test = tests, entry_point = entry_points)
/// ```
pub struct RewardEvaluator {
    config: EvaluatorConfig,
}

impl RewardEvaluator {
    pub fn new(config: EvaluatorConfig) -> Result<Self> {
        config.validate()?;

        if let Some(num_threads) = config.num_threads {
            ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .ok();
        }

        Ok(Self { config })
    }

    /// Check if text has valid `<think>...</think>` and `<answer>...</answer>` format.
    ///
    /// This validates that the model followed the structured reasoning format
    /// required for code generation tasks.
    fn has_valid_format(text: &str) -> bool {
        static THINK_PATTERN: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"(?is)<think>.*?</think>").unwrap());
        static ANSWER_PATTERN: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"(?is)<answer>.*?</answer>").unwrap());

        THINK_PATTERN.is_match(text) && ANSWER_PATTERN.is_match(text)
    }

    /// Evaluate format compliance for a batch of LLM outputs.
    ///
    /// Returns 1.0 for properly formatted outputs (with both `<think>` and `<answer>` tags),
    /// 0.0 otherwise.
    pub fn evaluate_response_format(&self, completions: &[String]) -> Vec<f64> {
        completions
            .iter()
            .map(|completion| {
                if Self::has_valid_format(completion) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Evaluate a single LLM output by executing the extracted code against tests.
    ///
    /// Returns 1.0 if all tests pass, 0.0 otherwise.
    fn evaluate_single_execution(&self, completion: &str, test: &str, entry_point: &str) -> f64 {
        if test.is_empty() || test == "null" {
            return 0.0;
        }

        let code = extract_code_from_completion(completion);
        if code.trim().is_empty() {
            return 0.0;
        }

        // Add standard typing imports
        let code_with_imports = format!(
            "from typing import List, Optional, Dict, Set, Tuple, Any\n\n{}",
            code
        );

        // Validate entry point exists in the generated code.
        //
        // The entry point specifies how the test code will call the solution:
        //
        // Example 1 - Simple function:
        //    entry_point: "add"
        //    generated code must contain: def add(...)
        //    test calls: add(1, 2)
        //
        // Example 2 - Class method:
        //     entry_point: "Solution().twoSum"
        //     generated code must contain: class Solution with def twoSum(...)
        //     test class: Solution().two_sum([1, 2], 3)
        //
        // This validation prevents false positives where the model generates code
        // but with wrong function/class names.
        if !entry_point.is_empty() && entry_point != "null" {
            // Extract method name: "Solution().twoSum" -> "twoSum", "add" -> "add"
            let method_name = if entry_point.contains('.') {
                entry_point.split('.').last().unwrap_or(entry_point)
            } else {
                entry_point
            };

            // Verify method/function definition exists
            if !code_with_imports.contains(&format!("def {}", method_name)) {
                return 0.0;
            }

            // For class-based entry points, verify the class exists
            if entry_point.contains("Solution().") && !code_with_imports.contains("class Solution")
            {
                return 0.0;
            }
        }

        // Wrap test code to run all tests
        let wrapped_tests = wrap_tests_for_complete_execution(test, entry_point);

        // Combine solution and tests
        let full_code = format!("{}\n\n{}", code_with_imports, wrapped_tests);

        // Execute in sandbox and return result
        match run_sandboxed_tests(
            &full_code,
            self.config.timeout_seconds,
            self.config.memory_limit_mb,
            self.config.cpu_time_limit,
        ) {
            Ok((all_passed, _tests_passed, _tests_total)) => {
                if all_passed {
                    1.0
                } else {
                    0.0
                }
            }
            Err(e) => {
                eprintln!("Execution error: {}", e);
                0.0
            }
        }
    }

    /// Evaluate sandboxed code execution for a batch in parallel.
    ///
    /// Uses Rayon to process completions (LLM outputs) in parallel across the thread pool.
    /// Each completion is evaluated independently with no shared state.
    ///
    /// # Arguments
    /// - `completions`: LLM outputs to evaluate
    /// - `tests`: Test code for each completion
    /// - `entry_points`: Function/method to test for each completion (e.g., "add" or "Solution().method")
    ///
    /// # Returns
    /// Vector of rewards (1.0 = all tests passed, 0.0 = failed or error)
    ///
    /// # Panics
    /// Panics if `completions`, `tests`, and `entry_points` have different lengths.
    pub fn evaluate_execution_batch(
        &self,
        completions: &[String],
        tests: &[String],
        entry_points: &[String],
    ) -> Vec<f64> {
        assert_eq!(
            completions.len(),
            tests.len(),
            "Completions and tests must have the same length"
        );
        assert_eq!(
            completions.len(),
            entry_points.len(),
            "Completions and entry_points must have same length"
        );

        completions
            .par_iter()
            .zip(tests.par_iter())
            .zip(entry_points.par_iter())
            .map(|((completion, test), entry_point)| {
                self.evaluate_single_execution(completion, test, entry_point)
            })
            .collect()
    }
}
