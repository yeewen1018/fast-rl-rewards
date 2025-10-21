//! src/bindings.rs
//!
//! Python bindings via PyO3.
//!
//! Provides two interfaces:
//! 1. Module-level functions - Simple API using a default RewardEvaluator
//! 2. RewardEvaluator class - Advanced API with custom configuration
//!
//! # Input Handling
//! Accepts completions in multiple formats for compatibility with various RL libraries:
//! - Direct strings: `["code1", "code2"]`
//! - Dicts with "content" key: `[{"content": "code1"}, ...]`
//! - Lists of dicts: `[[{"content": "code1"}], ...]`
//!
//! This flexibility allows drop-in replacement in TRL, Ray RLlib, and custom workflows.

use crate::evaluator::{EvaluatorConfig, RewardEvaluator};
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// ==========================================================================================

/// Global default evaluator for module-level functions.
///
/// Uses default configuration (32 threads for parallelism, 15s timeout, 512MB memory limit
/// for sandbox execution). Initialized lazily on first use.
static DEFAULT_EVALUATOR: Lazy<RewardEvaluator> = Lazy::new(|| {
    RewardEvaluator::new(EvaluatorConfig::default())
        .expect("Default evaluator configuration should always be valid")
});

// ==========================================================================================

/// Python-facing reward evaluator class
///
/// Provides full control over evaluation configuration including timeouts,
/// memory limits, and thread count.
///
/// # Examples
/// ```python
/// from fastrlrewards import RewardEvaluator
///
/// evaluator = RewardEvaluator(
///     timeout_seconds = 20,
///     memory_limit_mb = 1024,
///     cpu_time_limit = 15,
///     num_threads = None,
/// )
///
/// format_scores = evaluator.format_reward(completions)
/// execution_scores = evaluator.execution_reward(
///     completions,
///     test = tests,
///     entry_point = entry_points
/// )
/// ```
#[pyclass(name = "RewardEvaluator")]
pub struct PyRewardEvaluator {
    evaluator: RewardEvaluator,
}

#[pymethods]
impl PyRewardEvaluator {
    #[new]
    #[pyo3(signature = (timeout_seconds=15, memory_limit_mb=512, cpu_time_limit=12, num_threads=32))]
    fn new(
        timeout_seconds: u64,
        memory_limit_mb: u64,
        cpu_time_limit: u64,
        num_threads: usize,
    ) -> PyResult<Self> {
        let config = EvaluatorConfig {
            timeout_seconds,
            memory_limit_mb,
            cpu_time_limit,
            num_threads: Some(num_threads),
        };

        let evaluator = RewardEvaluator::new(config)
            .map_err(|e| PyValueError::new_err(format!("Invalid configuration: {}", e)))?;

        Ok(Self { evaluator })
    }

    /// Evaluate format compliance of LLM outputs (checks for `<think>` and `<answer>` tags).
    ///
    /// Returns 1.0 for completions with valid format, 0.0 otherwise.
    ///
    /// # Arguments:
    /// - `completions`: List of completion strings/dicts
    ///
    /// # Returns
    /// List of floats (1.0 or 0.0)
    fn format_reward(&self, completions: &Bound<'_, PyList>) -> PyResult<Vec<f64>> {
        let completions = extract_completions_from_pylist(completions)?;
        Ok(self.evaluator.evaluate_response_format(&completions))
    }

    /// Evaluate execution rewards (runs code with tests).
    ///
    /// Executes code in sandboxed environment and returns rewards based on
    /// whether all tests passed.
    ///
    /// # Arguments:
    /// - `completions`: List of LLM outputs
    /// - `kwargs["test"]`: List of test code strings
    /// - `kwargs["entry_point"]`: List of entry points (e.g., "add" or "Solution().method")
    ///
    /// # Returns
    /// List of floats (1.0 = all tests passed, 0.0 = failed/error)
    #[pyo3(signature = (completions, **kwargs))]
    fn execution_reward(
        &self,
        py: Python,
        completions: &Bound<'_, PyList>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<f64>> {
        let completions = extract_completions_from_pylist(completions)?;

        let (tests, entry_points) = if let Some(kwargs) = kwargs {
            let tests = extract_string_list_from_kwargs(kwargs, "test", completions.len())?;
            let entry_points =
                extract_string_list_from_kwargs(kwargs, "entry_point", completions.len())?;
            (tests, entry_points)
        } else {
            (
                vec![String::new(); completions.len()],
                vec![String::new(); completions.len()],
            )
        };

        py.detach(|| {
            Ok(self
                .evaluator
                .evaluate_execution_batch(&completions, &tests, &entry_points))
        })
    }
}

// ==========================================================================================

/// Module-level function for format reward (uses default evaluator)
///
/// Convenience function for simple use cases. Uses global default evaluator
/// with standard configuration.
///
/// # Examples
/// ```python
/// from fastrlrewards import format_reward
///
/// scores = format_reward(completions)
/// ```
#[pyfunction]
pub fn format_reward(completions: &Bound<'_, PyList>) -> PyResult<Vec<f64>> {
    let completions = extract_completions_from_pylist(completions)?;
    Ok(DEFAULT_EVALUATOR.evaluate_response_format(&completions))
}

/// Module-level function for execution reward (uses default evaluator).
///
/// Convenience function for simple use cases. Uses global default evaluator
/// with standard configuration.
///
/// # Examples
/// ```python
/// from fastrlrewards import execution_reward
///
/// scores = execution_reward(completions, test=tests, entry_point=entry_points)
/// ```
#[pyfunction]
#[pyo3(signature = (completions, **kwargs))]
pub fn execution_reward(
    py: Python,
    completions: &Bound<'_, PyList>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<f64>> {
    let completions = extract_completions_from_pylist(completions)?;

    let (tests, entry_points) = if let Some(kwargs) = kwargs {
        let tests = extract_string_list_from_kwargs(kwargs, "test", completions.len())?;
        let entry_points =
            extract_string_list_from_kwargs(kwargs, "entry_point", completions.len())?;
        (tests, entry_points)
    } else {
        (
            vec![String::new(); completions.len()],
            vec![String::new(); completions.len()],
        )
    };

    py.detach(|| {
        Ok(DEFAULT_EVALUATOR.evaluate_execution_batch(&completions, &tests, &entry_points))
    })
}

// ==========================================================================================

/// Helper function to extract completions from various Python input formats:
///
/// - Direct strings: `["code1", "code2"]` (Ray RLlib)
/// - Dicts with "content": `[{"content": "code1"}]` (TRL)
/// - Lists of dicts: `[[{"content": "code1"}]]` (some TRL versions)
/// - Fallback to string conversion
fn extract_completions_from_pylist(completions: &Bound<'_, PyList>) -> PyResult<Vec<String>> {
    let mut result = Vec::with_capacity(completions.len());

    for item in completions.iter() {
        let text = if let Ok(s) = item.extract::<String>() {
            // Case 1: Direct string
            s
        } else if let Ok(dict) = item.downcast::<PyDict>() {
            // Case 2: Dictionary with "content" key
            dict.get_item("content")?
                .and_then(|value| value.extract::<String>().ok())
                .unwrap_or_default()
        } else if let Ok(list) = item.downcast::<PyList>() {
            // Case 3: List of dicts (take first element)
            if !list.is_empty() {
                if let Ok(first) = list.get_item(0) {
                    if let Ok(dict) = first.downcast::<PyDict>() {
                        // First element is a dict - extract "content"
                        dict.get_item("content")?
                            .and_then(|value| value.extract::<String>().ok())
                            .unwrap_or_default()
                    } else {
                        // First element is not a dict - convert to string
                        first.str()?.to_string()
                    }
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        } else {
            // Case 4: Fallback - convert to string
            item.str()?.to_string()
        };

        result.push(text);
    }

    Ok(result)
}

/// Helper function to extract string lists from kwargs (for test= and entry_point= arguments)
///
/// # Errors
/// Returns an error if the provided list length does not match the expected length
fn extract_string_list_from_kwargs(
    kwargs: &Bound<'_, PyDict>,
    key: &str,
    expected_len: usize,
) -> PyResult<Vec<String>> {
    if let Some(value) = kwargs.get_item(key)? {
        if let Ok(list) = value.downcast::<PyList>() {
            let mut result = Vec::with_capacity(list.len());
            for item in list.iter() {
                result.push(item.extract::<String>().unwrap_or_default());
            }

            // Validate length
            if result.len() != expected_len {
                return Err(PyValueError::new_err(format!(
                    "Length mismatch: {} has {} items but expected {} (same as completions)",
                    key,
                    result.len(),
                    expected_len
                )));
            }

            return Ok(result);
        }
    }

    // Key not found - return empty strings (allow missing kwargs entirely)
    Ok(vec![String::new(); expected_len])
}
