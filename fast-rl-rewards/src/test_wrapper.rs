//! src/test_wrapper.rs

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use regex::Regex;

static ASSERT_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\s*)(assert\s+.+)").unwrap());

static CHECK_DEF_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"def\s+check\s*\(").unwrap());

static INDENT_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\s*)").unwrap());

#[pyfunction]
pub fn wrap_tests_for_complete_execution(test_code: &str, entry_point: &str) -> String {
    // Check if there are any assertions
    if !ASSERT_PATTERN.is_match(test_code) {
        return test_code.to_string();
    }

    let lines: Vec<&str> = test_code.split('\n').collect();
    let assert_count = ASSERT_PATTERN.find_iter(test_code).count();
    let mut wrapped_lines: Vec<String> = Vec::with_capacity(lines.len() + assert_count * 4 + 10);
    let mut in_check_function = false;
    let mut check_function_indent = String::new();

    for line in lines {
        // Detect check function definition
        if CHECK_DEF_PATTERN.is_match(line) {
            in_check_function = true;

            // Extract indent (match Python's re.match(r'(\s*)', line).group(1))
            if let Some(caps) = INDENT_PATTERN.captures(line) {
                check_function_indent = caps[1].to_string();
            }

            wrapped_lines.push(line.to_string());
            wrapped_lines.push(format!("{}    _results = []", check_function_indent));
            continue;
        }

        // Check if this line is an assertion inside check function
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

        // Detect end of check function
        if in_check_function {
            let trimmed = line.trim();

            let function_ended = trimmed.is_empty()
                || (!trimmed.is_empty()
                    && !line.starts_with(&format!("{} ", check_function_indent))
                    && !line.starts_with(&format!("{}\t", check_function_indent)));

            if function_ended {
                wrapped_lines.push(format!("{}    return _results", check_function_indent));
                wrapped_lines.push(String::new());
                in_check_function = false;

                // Add the current line if it is not empty
                if !trimmed.is_empty() {
                    wrapped_lines.push(line.to_string());
                }
                continue;
            }
        }

        wrapped_lines.push(line.to_string());
    }

    // If function never ended, add return
    if in_check_function {
        wrapped_lines.push(format!("{}    return _results", check_function_indent));
        wrapped_lines.push(String::new());
    }

    // Generate execution code
    wrapped_lines.push(format!("_test_results = check({})", entry_point));
    wrapped_lines.push(String::new());
    wrapped_lines.push("# Report test results".to_string());
    wrapped_lines.push("_passed = sum(_test_results)".to_string());
    wrapped_lines.push("_total = len(_test_results)".to_string());
    wrapped_lines.push(r#"print(f"TESTS_PASSED:{_passed}/{_total}")"#.to_string());
    wrapped_lines.push("exit(0 if _passed == _total else 1)".to_string());

    wrapped_lines.join("\n")
}
