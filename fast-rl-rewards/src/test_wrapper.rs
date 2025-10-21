//! src/test_wrapper.rs
//!
//! Test code transformation to run all tests instead of fail-fast:
//!
//! # Example
//! ```python
//! # Original:
//! def check(candidate):
//!     assert candidate(1, 2) == 3
//!     assert candidate(0, 0) == 0
//!
//! # Transformed:
//! def check(candidate):
//!     _results = []
//!     try:
//!         assert candidate(1, 2) == 3
//!         _results.append(True)
//!     except:
//!         _results.append(False)
//!     try:
//!         assert candidate(0, 0) == 0
//!         _results.append(True)
//!     except:
//!         _results.append(False)
//!     return _results
//!
//! _test_results = check(add)
//! _passed = sum(_test_results)
//! _total = len(_test_results)
//! print(f"TEST_PASSED:{_passed}/{_total}")
//! exit(0 if _passed == _total else 1)
//! ```

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use regex::Regex;

static ASSERT_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\s*)(assert\s+.+)").unwrap());
static CHECK_DEF_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"def\s+check\s*\(").unwrap());
static INDENT_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\s*)").unwrap());

/// # Arguments:
/// - `test_code`: Original test function (usually "def check(candidate): ...")
/// - `entry_point`: How to call the function (e.g., "add" or "Solution().method")
///
/// # Returns:
/// Transformed test code that runs all tests and prints "TEST_PASSED:X/Y"
#[pyfunction]
pub fn wrap_tests_for_complete_execution(test_code: &str, entry_point: &str) -> String {
    // Early return if no assertions to wrap
    if !ASSERT_PATTERN.is_match(test_code) {
        return test_code.to_string();
    }

    let lines: Vec<&str> = test_code.split('\n').collect();
    let assert_count = ASSERT_PATTERN.find_iter(test_code).count();

    // Pre-allocate capacity for better performance.
    //
    // Original:
    //   assert candidate(1, 2) == 3         # 1 line
    //
    // Wrapped:
    //   try:                                # +1
    //       assert candidate(1, 2) == 3     # (replaces original)
    //       _results.append(True)           # +1
    //   except:                             # +1
    //       _results.append(False)          # +1
    //   Total: +4 lines per assertion
    //
    // Additional overhead: ~10 lines for initialization, return, and reporting code
    let mut wrapped_lines: Vec<String> = Vec::with_capacity(lines.len() + assert_count * 4 + 10);
    let mut in_check_function = false;
    let mut check_function_indent = String::new();

    for line in lines {
        // 1. Detect check function definition
        if CHECK_DEF_PATTERN.is_match(line) {
            in_check_function = true;

            // Extract indentation level
            if let Some(caps) = INDENT_PATTERN.captures(line) {
                check_function_indent = caps[1].to_string();
            }

            wrapped_lines.push(line.to_string());
            wrapped_lines.push(format!("{}    _results = []", check_function_indent));
            continue;
        }

        // 2. Wrap assertions in try/except blocks
        if let Some(caps) = ASSERT_PATTERN.captures(line) {
            if in_check_function {
                let indent = &caps[1];
                let assertion = &caps[2];

                wrapped_lines.push(format!("{}try:", indent));
                wrapped_lines.push(format!("{}    {}", indent, assertion));
                wrapped_lines.push(format!("{}    _results.append(True)", indent));
                wrapped_lines.push(format!("{}except:", indent));
                wrapped_lines.push(format!("{}    _results.append(False)", indent));
                continue;
            }
        }

        // 3. Detect end of check function (dedent or empty line)
        if in_check_function {
            let trimmed = line.trim();

            // Function ends when we dedent or hit empty line
            let function_ended = trimmed.is_empty()
                || (!trimmed.is_empty()
                    && !line.starts_with(&format!("{} ", check_function_indent))
                    && !line.starts_with(&format!("{}\t", check_function_indent)));

            if function_ended {
                // Add return statement before exiting function
                wrapped_lines.push(format!("{}    return _results", check_function_indent));
                wrapped_lines.push(String::new());
                in_check_function = false;

                // Preserve current line if not empty
                if !trimmed.is_empty() {
                    wrapped_lines.push(line.to_string());
                }
                continue;
            }
        }

        // Regular line - pass through unchanged
        wrapped_lines.push(line.to_string());
    }

    // If function never explicitly ended, close it
    if in_check_function {
        wrapped_lines.push(format!("{}    return _results", check_function_indent));
        wrapped_lines.push(String::new());
    }

    // 4. Add execution and reporting code
    wrapped_lines.push(format!("_test_results = check({})", entry_point));
    wrapped_lines.push(String::new());
    wrapped_lines.push("# Report test results".to_string());
    wrapped_lines.push("_passed = sum(_test_results)".to_string());
    wrapped_lines.push("_total = len(_test_results)".to_string());
    wrapped_lines.push(r#"print(f"TESTS_PASSED:{_passed}/{_total}")"#.to_string());
    wrapped_lines.push("exit(0 if _passed == _total else 1)".to_string());

    wrapped_lines.join("\n")
}
