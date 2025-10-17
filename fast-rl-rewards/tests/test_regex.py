import fastrlrewards

test_completions = [
    "<think>Let me solve this.</think>\n<answer>def add(a, b): return a + b</answer>",
    "<answer>def multiply(x, y): return x * y</answer>",
    "no answer tags here",
]

expected_substrings = [
    "def add(a, b): return a + b",
    "def multiply(x, y): return x * y",
    "no answer tags here",
]

for i, completion in enumerate(test_completions):
    extracted = fastrlrewards.extract_code(completion)
    print(f"Input: {i+1}: {completion}")
    print(f"Extracted: {extracted}\n")
    assert expected_substrings[i] in extracted, f"Test {i+1} failed!"

print("pyo3 - extract_code works correctly!")