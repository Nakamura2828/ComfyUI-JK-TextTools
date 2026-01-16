"""
Tests for String Splitter Node

Run with: python test_string_splitter.py
"""

import inspect
from string_splitter import StringSplitter


def test_basic_split():
    """Test basic string splitting"""
    node = StringSplitter()
    
    result, count = node.split_string("a,b,c", ",")
    # With OUTPUT_IS_LIST, the function still returns a regular list
    # ComfyUI handles the list wrapping/unwrapping
    assert isinstance(result, list), f"Should return list, got {type(result)}"
    assert result == ["a", "b", "c"], f"Expected ['a', 'b', 'c'], got {result}"
    assert count == 3, f"Expected count 3, got {count}"
    
    print("✓ test_basic_split passed")


def test_frame_numbers():
    """Test splitting frame numbers (your use case)"""
    node = StringSplitter()
    
    result, count = node.split_string("10,25,42,100", ",")
    assert result == ["10", "25", "42", "100"], f"Expected frame list, got {result}"
    assert count == 4, f"Expected 4 frames, got {count}"
    
    print("✓ test_frame_numbers passed")


def test_whitespace_handling():
    """Test whitespace stripping"""
    node = StringSplitter()
    
    # With strip (default)
    result, _ = node.split_string(" a , b , c ", ",", strip_whitespace=True)
    assert result == ["a", "b", "c"], f"Expected stripped, got {result}"
    
    # Without strip
    result, _ = node.split_string(" a , b , c ", ",", strip_whitespace=False)
    assert result == [" a ", " b ", " c "], f"Expected unstripped, got {result}"
    
    print("✓ test_whitespace_handling passed")


def test_different_delimiters():
    """Test various delimiters"""
    node = StringSplitter()
    
    # Pipe
    result, _ = node.split_string("a|b|c", "|")
    assert result == ["a", "b", "c"], f"Pipe delimiter failed"
    
    # Newline
    result, _ = node.split_string("a\nb\nc", "\n")
    assert result == ["a", "b", "c"], f"Newline delimiter failed"
    
    # Multi-character
    result, _ = node.split_string("a::b::c", "::")
    assert result == ["a", "b", "c"], f"Multi-char delimiter failed"
    
    print("✓ test_different_delimiters passed")


def test_remove_empty():
    """Test removing empty strings"""
    node = StringSplitter()
    
    # With empty strings, keep them
    result, count = node.split_string("a,,c", ",", remove_empty=False)
    assert result == ["a", "", "c"], f"Expected empty string, got {result}"
    assert count == 3
    
    # Remove empty strings
    result, count = node.split_string("a,,c", ",", remove_empty=True)
    assert result == ["a", "c"], f"Expected no empty strings, got {result}"
    assert count == 2
    
    # Multiple consecutive delimiters
    result, count = node.split_string("a,,,b", ",", remove_empty=True)
    assert result == ["a", "b"], f"Expected ['a', 'b'], got {result}"
    assert count == 2
    
    print("✓ test_remove_empty passed")


def test_edge_cases():
    """Test edge cases"""
    node = StringSplitter()
    
    # Empty string
    result, count = node.split_string("", ",")
    assert result == [""], f"Empty string should split to list with one empty string"
    assert count == 1
    
    # Empty string with remove_empty
    result, count = node.split_string("", ",", remove_empty=True)
    assert result == [], f"Empty string with remove_empty should be empty list"
    assert count == 0
    
    # Single item
    result, count = node.split_string("only", ",")
    assert result == ["only"], f"Single item failed"
    assert count == 1
    
    # No delimiters
    result, count = node.split_string("no-delimiters", ",")
    assert result == ["no-delimiters"], f"No delimiters should return single item"
    assert count == 1
    
    print("✓ test_edge_cases passed")


def test_return_types():
    """Validate return types"""
    node = StringSplitter()
    
    result = node.split_string("a,b,c", ",")
    
    assert isinstance(result, tuple), f"Should return tuple"
    assert len(result) == 2, f"Should return 2 items"
    assert isinstance(result[0], list), f"First item should be list"
    assert isinstance(result[1], int), f"Second item should be int"
    
    print("✓ test_return_types passed")


def test_input_types_structure():
    """Validate INPUT_TYPES matches function signature"""
    input_types = StringSplitter.INPUT_TYPES()
    
    all_inputs = set()
    if "required" in input_types:
        all_inputs.update(input_types["required"].keys())
    if "optional" in input_types:
        all_inputs.update(input_types["optional"].keys())
    
    function = getattr(StringSplitter(), StringSplitter.FUNCTION)
    sig = inspect.signature(function)
    function_params = set(sig.parameters.keys()) - {'self'}
    
    missing = function_params - all_inputs
    extra = all_inputs - function_params
    
    assert not missing, f"Function has params not in INPUT_TYPES: {missing}"
    assert not extra, f"INPUT_TYPES has entries not in function: {extra}"
    
    print("✓ test_input_types_structure passed")


def run_all_tests():
    """Run all test functions"""
    print("Running tests for StringSplitter...\n")
    
    try:
        test_basic_split()
        test_frame_numbers()
        test_whitespace_handling()
        test_different_delimiters()
        test_remove_empty()
        test_edge_cases()
        test_return_types()
        test_input_types_structure()
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("="*50)
        return True
        
    except AssertionError as e:
        print("\n" + "="*50)
        print(f"❌ TEST FAILED: {e}")
        print("="*50)
        return False
    except Exception as e:
        print("\n" + "="*50)
        print(f"❌ UNEXPECTED ERROR: {e}")
        print("="*50)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
