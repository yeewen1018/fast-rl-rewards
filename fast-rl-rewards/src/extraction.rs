//! src/code_extractor.rs
//!
//! Code extraction from structured LLM responses.
//!
//! This module handles extracting Python code from completions that follow the
//! `<think>...</think>` `<answer>...</answer>` format used in reasoning models.
//!
//! # Extraction strategy:
//! 1. Try to extract from `<answer>...</answer>` tags
//! 2. Fallback to markdown code blocks (```python```)
//! 3. Return entire text as last resort.
//!
//! Markdown fences inside answer tags are automatically stripped.
//!
//! # Examples
//! ```python
//! import fastrlrewards
//!
//! completion = "<think>reasoning</think>\n<answer>```python\nprint('hi')\n```</answer>"
//! code = fastrlrewards.extract_code_from_completion(completion)
//! assert code == "print('hi')"
//! ```

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use regex::Regex;

// Regex pattern for content within <answer>...</answer> tags (case-insensitive)
static ANSWER_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<answer>(.*?)</answer>").unwrap());

// Regex pattern for markdown code blocks with Python language specifier
static CODE_BLOCK_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)```python\s*\n(.*?)\n```").unwrap());

// Patterns for cleaning markdown code blocks inside answer tags
static MARKDOWN_START_PYTHON: Lazy<Regex> = Lazy::new(|| Regex::new(r"^```python\s*\n").unwrap());
static MARKDOWN_START_PLAIN: Lazy<Regex> = Lazy::new(|| Regex::new(r"^```\s*\n").unwrap());
static MARKDOWN_END: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n```\s*$").unwrap());

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
