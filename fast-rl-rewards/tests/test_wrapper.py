import fastrlrewards 
import re 
from datasets import load_dataset 

# Python reference implementation 
def wrap_tests_for_complete_execution_python(test_code: str, entry_point: str) -> str:
    """Python reference implementation"""
    assert_pattern = r'(\s*)(assert\s+.+)'
    assertions = re.findall(assert_pattern, test_code)
    
    if not assertions:
        return test_code
    
    lines = test_code.split('\n')
    wrapped_lines = []
    in_check_function = False
    check_function_indent = ""
    
    for line in lines:
        if re.match(r'def\s+check\s*\(', line):
            in_check_function = True
            check_function_indent = re.match(r'(\s*)', line).group(1)
            wrapped_lines.append(line)
            wrapped_lines.append(f"{check_function_indent}    _results = []")
            continue
        
        assert_match = re.match(r'(\s*)(assert\s+.+)', line)
        if assert_match and in_check_function:
            indent = assert_match.group(1)
            assertion = assert_match.group(2)
            
            wrapped_lines.append(f"{indent}try:")
            wrapped_lines.append(f"{indent}    {assertion}")
            wrapped_lines.append(f"{indent}    _results.append(True)")
            wrapped_lines.append(f"{indent}except:")
            wrapped_lines.append(f"{indent}    _results.append(False)")
            continue
        
        if in_check_function:
            if line.strip() == "" or (line.strip() and not line.startswith(check_function_indent + ' ') and not line.startswith(check_function_indent + '\t')):
                wrapped_lines.append(f"{check_function_indent}    return _results")
                wrapped_lines.append("")
                in_check_function = False
                
                if line.strip():
                    wrapped_lines.append(line)
                continue
        
        wrapped_lines.append(line)
    
    if in_check_function:
        wrapped_lines.append(f"{check_function_indent}    return _results")
        wrapped_lines.append("")
    
    wrapped_lines.append(f"_test_results = check({entry_point})")
    wrapped_lines.append("")
    wrapped_lines.append("# Report test results")
    wrapped_lines.append("_passed = sum(_test_results)")
    wrapped_lines.append("_total = len(_test_results)")
    wrapped_lines.append('print(f"TESTS_PASSED:{_passed}/{_total}")')
    wrapped_lines.append("exit(0 if _passed == _total else 1)")
    
    return '\n'.join(wrapped_lines)

def test_with_dataset(num_samples=100, show_failures_only=True, verbose=False):
    """
    Test wrapping function with real Code-R1 dataset
    
    Args:
        num_samples: Number of samples to test (None for all)
        show_failures_only: Only print details for failures
        verbose: Show detailed output for each test
    """
    print("\n" + "="*80)
    print("TESTING WITH CODE-R1 DATASET")
    print("="*80 + "\n")
    
    # Load dataset
    print("Loading Code-R1 dataset...")
    dataset = load_dataset("ganler/code-r1-12k", split="train")
    print(f"Loaded {len(dataset)} problems")
    
    # Filter valid samples
    print("\nFiltering samples with valid tests...")
    valid_dataset = dataset.filter(
        lambda x: x['test'] is not None 
        and x['test'] != 'null' 
        and x['entry_point'] is not None
        and x['entry_point'] != 'null'
    )
    print(f"Valid samples: {len(valid_dataset)}")
    
    # Sample if needed
    if num_samples and num_samples < len(valid_dataset):
        test_dataset = valid_dataset.shuffle(seed=42).select(range(num_samples))
        print(f"Testing with {num_samples} random samples\n")
    else:
        test_dataset = valid_dataset
        print(f"Testing with all {len(valid_dataset)} samples\n")
    
    # Run tests
    passed = 0
    failed = 0
    failures = []
    
    print("Running tests...")
    for idx, sample in enumerate(test_dataset):
        test_code = sample['test']
        entry_point = sample['entry_point']
        
        # Run both implementations
        try:
            python_output = wrap_tests_for_complete_execution_python(test_code, entry_point)
            rust_output = fastrlrewards.wrap_tests_for_complete_execution(test_code, entry_point)
            
            # Compare
            if python_output == rust_output:
                passed += 1
                if verbose and not show_failures_only:
                    print(f"✓ Sample {idx}: PASS")
            else:
                failed += 1
                failures.append({
                    'idx': idx,
                    'test_code': test_code,
                    'entry_point': entry_point,
                    'python_output': python_output,
                    'rust_output': rust_output
                })
                if show_failures_only or verbose:
                    print(f"✗ Sample {idx}: FAIL")
        
        except Exception as e:
            failed += 1
            failures.append({
                'idx': idx,
                'test_code': test_code,
                'entry_point': entry_point,
                'error': str(e)
            })
            if show_failures_only or verbose:
                print(f"✗ Sample {idx}: ERROR - {e}")
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(test_dataset)} ({passed} passed, {failed} failed)")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(test_dataset)}")
    print(f"Passed: {passed} ({passed/len(test_dataset)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(test_dataset)*100:.1f}%)")
    print("="*80 + "\n")
    
    # Show failure details
    if failures:
        print("\n" + "="*80)
        print(f"FAILURE DETAILS ({len(failures)} failures)")
        print("="*80 + "\n")
        
        for i, failure in enumerate(failures[:10]):  # Show first 10 failures
            print(f"{'='*80}")
            print(f"FAILURE {i+1}: Sample {failure['idx']}")
            print(f"{'='*80}")
            
            if 'error' in failure:
                print(f"ERROR: {failure['error']}")
                print(f"\nTest code (first 300 chars):")
                print(failure['test_code'][:300])
                print(f"\nEntry point: {failure['entry_point']}")
            else:
                print(f"Entry point: {failure['entry_point']}")
                print(f"\nTest code (first 300 chars):")
                print(failure['test_code'][:300])
                
                print(f"\n{'='*40}")
                print("PYTHON OUTPUT (first 500 chars):")
                print(f"{'='*40}")
                print(failure['python_output'][:500])
                
                print(f"\n{'='*40}")
                print("RUST OUTPUT (first 500 chars):")
                print(f"{'='*40}")
                print(failure['rust_output'][:500])
                
                # Show line-by-line diff for first mismatch
                py_lines = failure['python_output'].split('\n')
                rs_lines = failure['rust_output'].split('\n')
                
                print(f"\n{'='*40}")
                print("FIRST DIFFERENCES:")
                print(f"{'='*40}")
                for j, (py_line, rs_line) in enumerate(zip(py_lines[:20], rs_lines[:20])):
                    if py_line != rs_line:
                        print(f"Line {j+1}:")
                        print(f"  Python: {repr(py_line)}")
                        print(f"  Rust:   {repr(rs_line)}")
                        break
            
            print()
        
        if len(failures) > 10:
            print(f"... and {len(failures) - 10} more failures (not shown)")
    
    return failed == 0


if __name__ == "__main__":
    success = test_with_dataset(num_samples=100)