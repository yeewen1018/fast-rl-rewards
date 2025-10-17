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

test_cases = [
    # ===== BASIC CASES =====
    # Case 1: Clean answer tags
    ("<answer>def foo(): return 1</answer>", 
     "def foo(): return 1"),
    
    # Case 2: Answer with markdown python block
    ("<answer>```python\ndef foo(): return 1\n```</answer>", 
     "def foo(): return 1"),
    
    # Case 3: Answer with plain markdown block
    ("<answer>```\ndef foo(): return 1\n```</answer>", 
     "def foo(): return 1"),
    
    # ===== WHITESPACE EDGE CASES =====
    # Case 4: Multiple spaces after ```python
    ("<answer>```python   \ndef foo(): return 1\n```</answer>", 
     "def foo(): return 1"),
    
    # Case 5: Tab after ```python
    ("<answer>```python\t\ndef foo(): return 1\n```</answer>", 
     "def foo(): return 1"),
    
    # Case 6: Trailing whitespace after closing ```
    ("<answer>```python\ndef foo(): return 1\n```  </answer>", 
     "def foo(): return 1"),
    
    # Case 7: Mixed whitespace (spaces + tabs)
    ("<answer>```python \t \ndef foo(): return 1\n``` \t</answer>", 
     "def foo(): return 1"),
    
    # Case 8: Leading/trailing whitespace in answer tags
    ("<answer>  \n  def foo(): return 1  \n  </answer>", 
     "def foo(): return 1"),
    
    # ===== CASE INSENSITIVITY =====
    # Case 9: Uppercase answer tags
    ("<ANSWER>def foo(): return 1</ANSWER>", 
     "def foo(): return 1"),
    
    # Case 10: Mixed case answer tags
    ("<Answer>def foo(): return 1</Answer>", 
     "def foo(): return 1"),
    
    # ===== EMPTY/MINIMAL CASES =====
    # Case 11: Empty answer tags
    ("<answer></answer>", 
     ""),
    
    # Case 12: Answer tags with only whitespace
    ("<answer>   \n   </answer>", 
     ""),
    
    # ===== FALLBACK CASES =====
    # Case 13: No answer tags, but has code block
    ("Some text\n```python\ndef bar(): pass\n```\nMore text", 
     "def bar(): pass"),
    
    # Case 14: No structured format
    ("just some plain text", 
     "just some plain text"),
    
    # Case 15: Incomplete answer tags (no closing) - returns whole string
    ("<answer>def foo(): return 1", 
     "<answer>def foo(): return 1"),
    
    # ===== FULL FORMAT =====
    # Case 16: Think and answer tags
    ("<think>reasoning</think>\n<answer>```python\nx = 1\n```</answer>", 
     "x = 1"),
    
    # ===== COMPLEX REAL-WORLD CASES =====
    # Case 17: Multiple answer blocks (should match first)
    ("<answer>first</answer><answer>second</answer>", 
     "first"),
    
    # Case 18: Answer with markdown that shouldn't be stripped
    # (no newline after ```python, so pattern shouldn't match)
    ("<answer>```python code here```</answer>", 
     "```python code here```"),
    
    # Case 19: Code containing backticks in strings
    ("<answer>```python\ncode = '```'\nprint(code)\n```</answer>", 
     "code = '```'\nprint(code)"),
    
    # Case 20: Multiline code with proper formatting
    ("<answer>```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```</answer>", 
     "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"),
    
    # Case 21: No markdown fence but still in answer tags
    ("<think>Let me solve this</think>\n<answer>x = 42\ny = x * 2</answer>", 
     "x = 42\ny = x * 2"),
]

for i, (input_text, expected) in enumerate(test_cases):
    extracted = fastrlrewards.extract_code_from_completion(input_text)
    assert extracted == expected, (
        f"Test case {i+1} failed!\n"
        f"Input: {input_text[:100]}...\n"
        f"Expected: {expected}\n"
        f"Got: {extracted}"
    )
    print(f"✓ Test case {i+1} passed")

print(f"\n✅ All {len(test_cases)} test cases passed!")