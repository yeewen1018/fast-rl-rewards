use once_cell::sync::Lazy;
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use regex::Regex;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::Duration;
use tempfile::Builder;
use wait_timeout::ChildExt;

/// Regex pattern to extract test results from output
static TEST_RESULTS_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"TESTS_PASSED:(\d+)/(\d+)").unwrap());

#[pyfunction]
#[pyo3(signature = (code, timeout=10, memory_limit_mb=512, cpu_time_limit=12))]
pub fn run_sandboxed_tests(
    code: &str,
    timeout: u64,
    memory_limit_mb: u64,
    cpu_time_limit: u64,
) -> PyResult<(bool, i32, i32)> {
    // Early return for empty code
    if code.trim().is_empty() {
        return Ok((false, 0, 0));
    }

    // Create temporary Python file in /tmp
    let mut temp_file = Builder::new()
        .suffix(".py")
        .tempfile_in("/tmp")
        .map_err(|e| PyErr::new::<PyIOError, _>(format!("Failed to create temp file: {}", e)))?;

    // Write code to temp file
    std::io::Write::write_all(&mut temp_file, code.as_bytes())
        .map_err(|e| PyErr::new::<PyIOError, _>(format!("Failed to write to temp file: {}", e)))?;

    // Get the file path
    let temp_path = temp_file.path();

    // Build firejail command
    let memory_limit_bytes = memory_limit_mb * 1_000_000;
    let mut cmd = Command::new("firejail");
    cmd.arg("--quiet")
        .arg("--private")
        .arg("--private-dev")
        .arg("--net=none")
        .arg("--x11=none")
        .arg("--nodbus")
        .arg(format!("--rlimit-as={}", memory_limit_bytes))
        .arg(format!("--rlimit-cpu={}", cpu_time_limit))
        .arg("--rlimit-nproc=10")
        .arg("--rlimit-fsize=10000000")
        .arg("python3")
        .arg("-u")
        .arg(temp_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .env("PYTHONPATH", "");

    // Spawn the process
    let mut child = cmd.spawn().map_err(|e| {
        PyErr::new::<PyRuntimeError, _>(format!(
            "Failed to spawn firejail process: {}. Is firejail installed?",
            e
        ))
    })?;

    // Take ownership of stdout
    let mut stdout = child.stdout.take().expect("Failed to take stdout");

    // Read stdout in a thread
    let stdout_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        stdout.read_to_end(&mut buf).ok();
        buf
    });

    // Wait with timeout
    let timeout_duration = Duration::from_secs(timeout);
    let status = match child
        .wait_timeout(timeout_duration)
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("Error waiting for process: {}", e)))?
    {
        Some(status) => status,
        None => {
            // Timeout - kill the process
            let _ = child.kill();
            let _ = child.wait();
            return Ok((false, 0, 0));
        }
    };

    // Get the output from thread
    let stdout_bytes = stdout_thread.join().expect("stdout thread panicked");
    let stdout_str = String::from_utf8_lossy(&stdout_bytes);
    let exit_code = status.code().unwrap_or(-1);

    // Parse test results from stdout
    let (tests_passed, tests_total) = TEST_RESULTS_PATTERN
        .captures(&stdout_str)
        .map(|caps| {
            let passed = caps[1].parse::<i32>().unwrap_or(0);
            let total = caps[2].parse::<i32>().unwrap_or(0);
            (passed, total)
        })
        .unwrap_or((0, 0));

    let all_passed = exit_code == 0 && tests_passed == tests_total && tests_total > 0;
    Ok((all_passed, tests_passed, tests_total))
}
