//! fast-rl-rewards
//!
//! This crate provides the functions to build a Rust-based, high-performance reward evaluation engine
//! for reinforcement learning workflows. It targets CPU-bound evaluation tasks like code execution,
//! mathematical verification, and symbolic computation.
//!
//! Uses Rayon for parallel execution and PyO3 for seamless Python integration.
//!
//! Note: The current version focuses on code generation tasks with structured reasoning format
//! (`<think>`/`<answer>` tags). The long-term goal is to provide UDF traits for custom reward functions,
//! where users can define reward logic in Python and this crate handles parallelization, sandboxing,
//! and aggregation.  
//!
//! # Quick Start
//! ```python
//! from fastrlrewards import execution_reward
//!
//! rewards = execution_reward(completions, test=tests, entry_point=entry_points)
//! ```
//!
//! # Modules
//!
//! - [`bindings`]: PyO3 Python interface
//! - [`evaluator`]: Core evaluation logic with Rayon parallelism
//! - [`extraction`]: Code extraction from structured responses
//! - [`test_wrapper`]: Test transformation for run-all-tests mode
//! - [`sandbox`]: Firejail sandboxed execution

mod bindings;
mod extraction;
mod test_wrapper;
mod evaluator;
mod sandbox;

use pyo3::prelude::*;

#[pymodule]
fn fastrlrewards(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Main evaluator class
    m.add_class::<bindings::PyRewardEvaluator>()?;

    // Convenience functions (module-level API using default PyRewardEvaluator)
    m.add_function(wrap_pyfunction!(bindings::format_reward, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::execution_reward, m)?)?;

    // Utility functions
    m.add_function(wrap_pyfunction!(
        extraction::extract_code_from_completion,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        test_wrapper::wrap_tests_for_complete_execution,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sandbox::execute_code_with_tests_firejail,
        m
    )?)?;
    Ok(())
}
