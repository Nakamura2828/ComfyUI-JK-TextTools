"""
Tests for String Index Selector Node

Run with: python test_string_index_selector.py
"""

import inspect
import sys
from pathlib import Path

# Add parent directory to path to import node modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from string_index_selector import StringIndexSelector


def test_basic_selection():
    """Test basic item selection"""
    node = StringIndexSelector()
    
    # Test selecting from comma-separated list
    result, count = node.select_by_index("a,b,c,d", ",", 0)
    assert result == "a", f"Expected 'a', got '{result}'"
    assert count == 4, f"Expected count 4, got {count}"
    
    result, count = node.select_by_index("a,b,c,d", ",", 2)
    assert result == "c", f"Expected 'c', got '{result}'"
    
    print("✓ test_basic_selection passed")


def test_frame_numbers():
    """Test your actual use case - frame numbers"""
    node = StringIndexSelector()
    
    frames = "10,25,42,100"
    
    # Loop iteration 0 → frame 10
    result, count = node.select_by_index(frames, ",", 0)
    assert result == "10", f"Expected '10', got '{result}'"
    assert count == 4, f"Expected 4 frames, got {count}"
    
    # Loop iteration 2 → frame 42
    result, count = node.select_by_index(frames, ",", 2)
    assert result == "42", f"Expected '42', got '{result}'"
    
    # Loop iteration 3 → frame 100
    result, count = node.select_by_index(frames, ",", 3)
    assert result == "100", f"Expected '100', got '{result}'"
    
    print("✓ test_frame_numbers passed")


def test_whitespace_handling():
    """Test whitespace stripping"""
    node = StringIndexSelector()
    
    # With whitespace, strip enabled (default)
    result, _ = node.select_by_index(" a , b , c ", ",", 1, strip_whitespace=True)
    assert result == "b", f"Expected 'b', got '{result}'"
    
    # With whitespace, strip disabled
    result, _ = node.select_by_index(" a , b , c ", ",", 1, strip_whitespace=False)
    assert result == " b ", f"Expected ' b ', got '{result}'"
    
    print("✓ test_whitespace_handling passed")


def test_different_delimiters():
    """Test various delimiter types"""
    node = StringIndexSelector()
    
    # Pipe delimiter
    result, _ = node.select_by_index("a|b|c", "|", 1)
    assert result == "b", f"Expected 'b', got '{result}'"
    
    # Tab delimiter
    result, _ = node.select_by_index("a\tb\tc", "\t", 2)
    assert result == "c", f"Expected 'c', got '{result}'"
    
    # Multi-character delimiter
    result, _ = node.select_by_index("a::b::c", "::", 0)
    assert result == "a", f"Expected 'a', got '{result}'"
    
    print("✓ test_different_delimiters passed")


def test_indexing_modes():
    """Test zero-indexed vs one-indexed"""
    node = StringIndexSelector()
    
    # Zero-indexed (0 = first item)
    result, _ = node.select_by_index("a,b,c", ",", 0, zero_indexed=True)
    assert result == "a", f"Expected 'a' with 0-indexing, got '{result}'"
    
    # One-indexed (1 = first item)
    result, _ = node.select_by_index("a,b,c", ",", 1, zero_indexed=False)
    assert result == "a", f"Expected 'a' with 1-indexing, got '{result}'"
    
    result, _ = node.select_by_index("a,b,c", ",", 2, zero_indexed=False)
    assert result == "b", f"Expected 'b' with 1-indexing, got '{result}'"
    
    print("✓ test_indexing_modes passed")


def test_out_of_range():
    """Test handling of invalid indices"""
    node = StringIndexSelector()
    
    # Index too high
    result, count = node.select_by_index("a,b,c", ",", 10)
    assert result == "", f"Expected empty string for out of range, got '{result}'"
    assert count == 3, f"Should still return correct count"
    
    # Negative index (if zero_indexed=True, shouldn't happen, but test anyway)
    result, count = node.select_by_index("a,b,c", ",", -1)
    assert result == "", f"Expected empty string for negative index, got '{result}'"
    
    print("✓ test_out_of_range passed")


def test_edge_cases():
    """Test edge cases"""
    node = StringIndexSelector()
    
    # Empty string
    result, count = node.select_by_index("", ",", 0)
    assert result == None, f"Expected empty for empty string"
    assert count == 0, f"Empty string splits to zero empty items"
    
    # Single item
    result, count = node.select_by_index("only", ",", 0)
    assert result == "only", f"Expected 'only', got '{result}'"
    assert count == 1
    
    # No delimiters found
    result, count = node.select_by_index("no-commas-here", ",", 0)
    assert result == "no-commas-here", f"Expected full string, got '{result}'"
    assert count == 1
    
    print("✓ test_edge_cases passed")


def test_return_types():
    """Validate return types match RETURN_TYPES"""
    node = StringIndexSelector()
    
    result = node.select_by_index("a,b,c", ",", 0)
    
    # Should return tuple
    assert isinstance(result, tuple), f"Should return tuple, got {type(result)}"
    assert len(result) == 2, f"Should return 2 items, got {len(result)}"
    
    # First item should be string
    assert isinstance(result[0], str), f"First return should be string, got {type(result[0])}"
    
    # Second item should be int
    assert isinstance(result[1], int), f"Second return should be int, got {type(result[1])}"
    
    print("✓ test_return_types passed")


def test_input_types_structure():
    """Validate INPUT_TYPES matches function signature"""
    input_types = StringIndexSelector.INPUT_TYPES()
    
    # Collect all inputs
    all_inputs = set()
    if "required" in input_types:
        all_inputs.update(input_types["required"].keys())
    if "optional" in input_types:
        all_inputs.update(input_types["optional"].keys())
    
    # Get function parameters
    function = getattr(StringIndexSelector(), StringIndexSelector.FUNCTION)
    sig = inspect.signature(function)
    function_params = set(sig.parameters.keys()) - {'self'}
    
    # Check they match
    missing = function_params - all_inputs
    extra = all_inputs - function_params
    
    assert not missing, f"Function has params not in INPUT_TYPES: {missing}"
    assert not extra, f"INPUT_TYPES has entries not in function: {extra}"
    
    print("✓ test_input_types_structure passed")

def test_output_type_handling():
    """Validate casting works vor various configured output_types"""
    node = StringIndexSelector()
    
    # default as string
    result, count = node.select_by_index("1,2,3", ",", 1)
    assert isinstance(result, str), f"Expected string, got '{type(result)}'"

    # casting to integer
    result, count = node.select_by_index("1,2,3", ",", 1, output_type='INT')
    assert isinstance(result, int), f"Expected integer, got '{type(result)}'"

    # casting to integer
    result, count = node.select_by_index("1,2,3", ",", 1, output_type='FLOAT')
    assert isinstance(result, float), f"Expected float 2, got '{type(result)}'"
    
    print("✓ test_output_type_handling passed")


def run_all_tests():
    """Run all test functions"""
    print("Running tests for StringIndexSelector...\n")
    
    try:
        test_basic_selection()
        test_frame_numbers()
        test_whitespace_handling()
        test_different_delimiters()
        test_indexing_modes()
        test_out_of_range()
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
