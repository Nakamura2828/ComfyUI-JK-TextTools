"""
Tests for String Splitter Node

Run with: python test_string_splitter.py
"""

import inspect
import sys
from pathlib import Path

# Add parent directory to path to import node modules
sys.path.insert(0, str(Path(__file__).parent.parent))
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
    assert result == [], f"Empty string should split to list with one empty string, saw {result}"
    assert count == 0, f"Empty string should have a count of zero, saw {result}"
    
    # Empty string with remove_empty
    result, count = node.split_string("", ",", remove_empty=True)
    assert result == [], f"Empty string with remove_empty should be empty list, saw {result}"
    assert count == 0, f"Empty string should have a count of zero, saw {result}"
    
    # Single item
    result, count = node.split_string("only", ",")
    assert result == ["only"], f"Single item failed"
    assert count == 1, f"Single item string should have a count of one, saw {result}"
    
    # No delimiters
    result, count = node.split_string("no-delimiters", ",")
    assert result == ["no-delimiters"], f"No delimiters should return single item"
    assert count == 1, f"List with no delimiters count of one, saw {result}"
    
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

"""
Additional tests for String Splitter type casting

Add these to your existing test_string_splitter.py
"""

def test_int_casting():
    """Test casting to integers"""
    node = StringSplitter()
    
    # Basic int casting
    result, count = node.split_string("10,25,42,100", ",", output_type="INT")
    assert result == [10, 25, 42, 100], f"Expected [10, 25, 42, 100], got {result}"
    assert all(isinstance(x, int) for x in result), "All items should be integers"
    
    # Negative numbers
    result, _ = node.split_string("-5,0,5", ",", output_type="INT")
    assert result == [-5, 0, 5], f"Expected [-5, 0, 5], got {result}"
    
    print("✓ test_int_casting passed")


def test_float_casting():
    """Test casting to floats"""
    node = StringSplitter()
    
    # Basic float casting
    result, count = node.split_string("1.5,2.0,3.14", ",", output_type="FLOAT")
    assert result == [1.5, 2.0, 3.14], f"Expected [1.5, 2.0, 3.14], got {result}"
    assert all(isinstance(x, float) for x in result), "All items should be floats"
    
    # Integers can be cast to float
    result, _ = node.split_string("10,20,30", ",", output_type="FLOAT")
    assert result == [10.0, 20.0, 30.0], f"Expected [10.0, 20.0, 30.0], got {result}"
    
    print("✓ test_float_casting passed")


def test_string_output_type():
    """Test explicit STRING output type"""
    node = StringSplitter()
    
    # Should remain strings even if they look like numbers
    result, _ = node.split_string("10,20,30", ",", output_type="STRING")
    assert result == ["10", "20", "30"], f"Expected ['10', '20', '30'], got {result}"
    assert all(isinstance(x, str) for x in result), "All items should be strings"
    
    print("✓ test_string_output_type passed")


def test_invalid_int_casting():
    """Test that invalid int casting raises proper error"""
    node = StringSplitter()
    
    try:
        node.split_string("10,abc,30", ",", output_type="INT")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Failed to convert" in str(e), "Error message should be helpful"
    
    print("✓ test_invalid_int_casting passed")


def test_invalid_float_casting():
    """Test that invalid float casting raises proper error"""
    node = StringSplitter()
    
    try:
        node.split_string("1.5,not_a_number,3.14", ",", output_type="FLOAT")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Failed to convert" in str(e), "Error message should be helpful"
    
    print("✓ test_invalid_float_casting passed")


def test_empty_string_with_casting():
    """Test empty string handling with type casting"""
    node = StringSplitter()
    
    # Empty strings should be removed before casting
    result, count = node.split_string("10,,20", ",", output_type="INT", remove_empty=True)
    assert result == [10, 20], f"Expected [10, 20], got {result}"
    assert count == 2
    
    # Without remove_empty, it should fail to cast empty string
    try:
        node.split_string("10,,20", ",", output_type="INT", remove_empty=False)
        assert False, "Should fail to cast empty string to INT"
    except ValueError:
        pass  # Expected
    
    print("✓ test_empty_string_with_casting passed")


def test_whitespace_with_casting():
    """Test whitespace handling with numeric casting"""
    node = StringSplitter()
    
    # Whitespace should be stripped before casting
    result, _ = node.split_string(" 10 , 20 , 30 ", ",", strip_whitespace=True, output_type="INT")
    assert result == [10, 20, 30], f"Expected [10, 20, 30], got {result}"
    
    print("✓ test_whitespace_with_casting passed")

def test_empty_string():
    node = StringSplitter()
    result, count = node.split_string("", ",")
    assert result == [], "Empty string should return empty list"
    assert count == 0

    print("✓ test_empty_string passed")

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

        test_int_casting()
        test_float_casting()
        test_string_output_type()
        test_invalid_int_casting()
        test_invalid_float_casting()
        test_empty_string_with_casting()
        test_whitespace_with_casting()

        test_empty_string()
        
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
