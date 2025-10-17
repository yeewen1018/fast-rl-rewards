mod code_extractor;
mod code_wrapper;

use pyo3::prelude::*;

#[pyfunction]
fn extract_code(completion: &str) -> String {
    if let Some(start) = completion.find("<answer>") {
        if let Some(end) = completion.find("</answer>") {
            return completion[start + 8..end].to_string();
        }
    }
    completion.to_string()
}

#[pymodule]
fn fastrlrewards(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_code, m)?)?;
    m.add_function(wrap_pyfunction!(
        code_extractor::extract_code_from_completion,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        code_wrapper::wrap_tests_for_complete_execution,
        m
    )?)?;
    Ok(())
}
