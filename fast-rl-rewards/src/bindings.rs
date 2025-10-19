use crate::reward_evaluator::{EvaluatorConfig, RewardEvaluator};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Global default evaluator for module-level functions
static DEFAULT_EVALUATOR: Lazy<RewardEvaluator> =
    Lazy::new(|| RewardEvaluator::new(EvaluatorConfig::default()));

/// Python-facing reward evaluator class
#[pyclass(name = "RewardEvaluator")]
pub struct PyRewardEvaluator {
    evaluator: RewardEvaluator,
}

#[pymethods]
impl PyRewardEvaluator {
    #[new]
    #[pyo3(signature = (timeout_seconds=15, memory_limit_mb=512, cpu_time_limit=12))]
    fn new(timeout_seconds: u64, memory_limit_mb: u64, cpu_time_limit: u64) -> Self {
        let config = EvaluatorConfig {
            timeout_seconds,
            memory_limit_mb,
            cpu_time_limit,
        };

        Self {
            evaluator: RewardEvaluator::new(config),
        }
    }

    /// Evaluate format rewards (checks for <think> and <answer> tags)
    fn format_reward(&self, completions: &Bound<'_, PyList>) -> PyResult<Vec<f64>> {
        let completions = extract_completions_from_pylist(completions)?;
        Ok(self.evaluator.evaluate_format(&completions))
    }

    /// Evaluate execution rewards (runs code with tests)
    /// Matches TRL's expected signature
    #[pyo3(signature = (completions, **kwargs))]
    fn execution_reward(
        &self,
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

        Ok(self
            .evaluator
            .evaluate_execution_batch(&completions, &tests, &entry_points))
    }
}

/// Helper function to extract completions from various input formats
fn extract_completions_from_pylist(completions: &Bound<'_, PyList>) -> PyResult<Vec<String>> {
    let mut result = Vec::with_capacity(completions.len());

    for item in completions.iter() {
        // Now .iter() works!
        let text = if let Ok(s) = item.extract::<String>() {
            // Direct string
            s
        } else if let Ok(dict) = item.downcast::<PyDict>() {
            // Dictionary with "content" key
            dict.get_item("content")?
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_default()
        } else if let Ok(list) = item.downcast::<PyList>() {
            // List of dicts (take first element)
            if !list.is_empty() {
                if let Ok(first) = list.get_item(0) {
                    if let Ok(dict) = first.downcast::<PyDict>() {
                        dict.get_item("content")?
                            .and_then(|v| v.extract::<String>().ok())
                            .unwrap_or_default()
                    } else {
                        // Not a dict, convert first element to string
                        first.str()?.to_string()
                    }
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        } else {
            // Fallback: convert to string
            item.str()?.to_string()
        };

        result.push(text);
    }

    Ok(result)
}

/// Helper to extract string lists from kwargs
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

            // Pad with empty strings if too short
            while result.len() < expected_len {
                result.push(String::new());
            }

            return Ok(result);
        }
    }

    // Key not found or not a list, return empty strings
    Ok(vec![String::new(); expected_len])
}

/// Module-level function for format reward (uses default evaluator)
#[pyfunction]
pub fn format_reward(completions: &Bound<'_, PyList>) -> PyResult<Vec<f64>> {
    let completions = extract_completions_from_pylist(completions)?;
    Ok(DEFAULT_EVALUATOR.evaluate_format(&completions))
}

/// Module-level function for execution reward (uses default evaluator)
#[pyfunction]
#[pyo3(signature = (completions, **kwargs))]
pub fn execution_reward(
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

    Ok(DEFAULT_EVALUATOR.evaluate_execution_batch(&completions, &tests, &entry_points))
}
