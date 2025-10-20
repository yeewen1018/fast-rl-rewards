//! src/code_extractor.rs
//!
//! Contains the functions to extract code from LLM responses that follow a structured
//! format with `<think>...</think>` and `<answer>`...`</answer>` tags.

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use regex::Regex;

/// Regex patterns:
/// 1. Matches content within <answer>...</answer> tags (case-insensitive)
static ANSWER_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<answer>(.*?)</answer>").unwrap());

/// 2. Matches markdown code blocks with Python language specifier
static CODE_BLOCK_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)```python\s*\n(.*?)\n```").unwrap());

/// The following regex patterns are for cleaning markdown code blocks inside answer tags.
/// 3. Matches opening markdown fence with python language: ```python followed by whitespace and newline
static MARKDOWN_START_PYTHON: Lazy<Regex> = Lazy::new(|| Regex::new(r"^```python\s*\n").unwrap());

/// 4. Matches opening markdown fence without language: ``` followed by whitespace and newline
static MARKDOWN_START_PLAIN: Lazy<Regex> = Lazy::new(|| Regex::new(r"^```\s*\n").unwrap());

/// 5. Matches closing markdown fence: newline followed by ``` and optional trailing whitespace
static MARKDOWN_END: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n```\s*$").unwrap());

/// Extracts Python code from a structured LLM completion. It tries multiple extraction strategies
/// in the following order:
/// 1. Extract from `<answer>..</answer>` tags
/// 2. Extract from markdown code blocks with `python` language specifier
/// 3. Return the entire completion as is (fallback for malformed responses)
///
/// When extracting from `<answer>..</answer>` tags, the function automatically strips any
/// markdown code block fences (```python or ```) that may wrap the code.
///
/// # Arguments:
/// - `completion`: The full text of the LLM completion to parse
///
/// # Returns:
/// - `String`: The extracted code
///
/// # Examples:
/// ```python
/// import fastrlrewards
///
/// let completion = "<think>reasoning here</think>\n<answer>```python\nprint('hello')\n```</answer>";
/// let code = fastrlrewards.extract_code_from_completion(completion);
/// assert_eq!(code, "print('hello')");
/// ```
#[pyfunction]
pub fn extract_code_from_completion(completion: &str) -> String {
    if let Some(captures) = ANSWER_PATTERN.captures(completion) {
        let code = captures[1].trim();

        let code = MARKDOWN_START_PYTHON.replace(code, "");
        let code = MARKDOWN_START_PLAIN.replace(&code, "");
        let code = MARKDOWN_END.replace(&code, "");

        return code.into_owned();
    }

    if let Some(captures) = CODE_BLOCK_PATTERN.captures(completion) {
        return captures[1].trim().to_string();
    }

    completion.trim().to_string()
}
