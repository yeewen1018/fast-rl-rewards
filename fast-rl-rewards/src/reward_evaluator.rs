use crate::code_extractor::extract_code_from_completion;
use crate::code_wrapper::wrap_tests_for_complete_execution;
use crate::sandbox::execute_code_with_tests_firejail;

/// Configuration for the reward evaluator
#[derive(Clone, Debug)]
pub struct EvaluatorConfig {
    pub timeout_seconds: u64,
    pub memory_limit_mb: u64,
    pub cpu_time_limit: u64,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: 15,
            memory_limit_mb: 512,
            cpu_time_limit: 12,
        }
    }
}

pub struct RewardEvaluator {
    config: EvaluatorConfig,
}

impl RewardEvaluator {
    pub fn new(config: EvaluatorConfig) -> Self {
        Self { config }
    }

    fn has_valid_format(text: &str) -> bool {
        use once_cell::sync::Lazy;
        use regex::Regex;

        static THINK_PATTERN: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"(?is)<think>.*?</think>").unwrap());
        static ANSWER_PATTERN: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"(?is)<answer>.*?</answer>").unwrap());

        THINK_PATTERN.is_match(text) && ANSWER_PATTERN.is_match(text)
    }

    pub fn evaluate_format(&self, completions: &[String]) -> Vec<f64> {
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

    pub fn evaluate_single(&self, completion: &str, test: &str, entry_point: &str) -> f64 {
        if test.is_empty() || test == "null" {
            return 0.0;
        }

        let code = extract_code_from_completion(completion);

        if code.trim().is_empty() {
            return 0.0;
        }

        let code_with_imports = format!(
            "from typing import List, Optional, Dict, Set, Tuple, Any\n\n{}",
            code
        );

        if !entry_point.is_empty() && entry_point != "null" {
            let method_name = if entry_point.contains('.') {
                entry_point.split('.').last().unwrap_or(entry_point)
            } else {
                entry_point
            };

            if !code_with_imports.contains(&format!("def {}", method_name)) {
                return 0.0;
            }

            if entry_point.contains("Solution().") && !code_with_imports.contains("class Solution")
            {
                return 0.0;
            }
        }

        let wrapped_tests = wrap_tests_for_complete_execution(test, entry_point);
        let full_code = format!("{}\n\n{}", code_with_imports, wrapped_tests);
        match execute_code_with_tests_firejail(
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

    /// Evaluate execution rewards for a batch (single-threaded for now)
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
            .iter()
            .zip(tests.iter())
            .zip(entry_points.iter())
            .map(|((c, t), e)| self.evaluate_single(c, t, e))
            .collect()
    }
}
