"""
Tests for String Joiner Node

Run with: python test_string_joiner.py
"""

import inspect
from string_joiner import StringJoiner


def test_basic_join():
    """Test basic string joining"""
    node = StringJoiner()
    
    # Basic join
    result, count = node.join_list(["a", "b", "c"], ",")
    assert result == "a,b,c", f"Expected 'a,b,c', got '{result}'"
    assert count == 3, f"Expected count 3, got {count}"
    
    print("✓ test_basic_join passed")


def test_with_string_splitter():
    """Test round-trip with String Splitter"""
    from string_splitter import StringSplitter
    
    splitter = StringSplitter()
    joiner = StringJoiner()
    
    # Split then join - should get back original
    original = "10,25,42,100"
    split_list, _ = splitter.split_string(original, ",")
    rejoined, _ = joiner.join_list(split_list, ",")
    
    assert rejoined == original, f"Expected '{original}', got '{rejoined}'"
    
    print("✓ test_with_string_splitter passed")


def test_integer_list():
    """Test joining list of integers"""
    node = StringJoiner()
    
    # Integer list (from String Splitter with INT output)
    int_list = [10, 25, 42, 100]
    result, count = node.join_list(int_list, ",")
    
    assert result == "10,25,42,100", f"Expected '10,25,42,100', got '{result}'"
    assert count == 4
    
    print("✓ test_integer_list passed")


def test_float_list():
    """Test joining list of floats"""
    node = StringJoiner()
    
    float_list = [1.5, 2.0, 3.14]
    result, _ = node.join_list(float_list, ",")
    
    assert result == "1.5,2.0,3.14", f"Expected '1.5,2.0,3.14', got '{result}'"
    
    print("✓ test_float_list passed")


def test_different_delimiters():
    """Test various delimiters"""
    node = StringJoiner()
    
    test_list = ["a", "b", "c"]
    
    # Pipe delimiter
    result, _ = node.join_list(test_list, "|")
    assert result == "a|b|c", f"Pipe delimiter failed: {result}"
    
    # Newline
    result, _ = node.join_list(test_list, "\n")
    assert result == "a\nb\nc", f"Newline delimiter failed: {result}"
    
    # Space
    result, _ = node.join_list(test_list, " ")
    assert result == "a b c", f"Space delimiter failed: {result}"
    
    # Multi-character
    result, _ = node.join_list(test_list, " :: ")
    assert result == "a :: b :: c", f"Multi-char delimiter failed: {result}"
    
    print("✓ test_different_delimiters passed")


def test_edge_cases():
    """Test edge cases"""
    node = StringJoiner()
    
    # Empty list
    result, count = node.join_list([], ",")
    assert result == "", f"Empty list should give empty string"
    assert count == 0
    
    # Single item
    result, count = node.join_list(["only"], ",")
    assert result == "only", f"Single item failed: {result}"
    assert count == 1
    
    # Empty strings in list
    result, _ = node.join_list(["a", "", "c"], ",")
    assert result == "a,,c", f"Empty string handling failed: {result}"
    
    print("✓ test_edge_cases passed")


def test_mixed_types():
    """Test list with mixed types"""
    node = StringJoiner()
    
    mixed_list = ["text", 42, 3.14, True, None]
    result, count = node.join_list(mixed_list, " | ")
    
    assert result == "text | 42 | 3.14 | True | None", f"Mixed types failed: {result}"
    assert count == 5
    
    print("✓ test_mixed_types passed")


def test_return_types():
    """Validate return types"""
    node = StringJoiner()
    
    result = node.join_list(["a", "b"], ",")
    
    assert isinstance(result, tuple), f"Should return tuple"
    assert len(result) == 2, f"Should return 2 items"
    assert isinstance(result[0], str), f"First item should be string"
    assert isinstance(result[1], int), f"Second item should be int"
    
    print("✓ test_return_types passed")


def test_input_types_structure():
    """Validate INPUT_TYPES matches function signature"""
    input_types = StringJoiner.INPUT_TYPES()
    
    all_inputs = set()
    if "required" in input_types:
        all_inputs.update(input_types["required"].keys())
    if "optional" in input_types:
        all_inputs.update(input_types["optional"].keys())
    
    function = getattr(StringJoiner(), StringJoiner.FUNCTION)
    sig = inspect.signature(function)
    function_params = set(sig.parameters.keys()) - {'self'}
    
    missing = function_params - all_inputs
    extra = all_inputs - function_params
    
    assert not missing, f"Function has params not in INPUT_TYPES: {missing}"
    assert not extra, f"INPUT_TYPES has entries not in function: {extra}"
    
    print("✓ test_input_types_structure passed")


def test_full_workflow():
    """Test complete workflow: split → process → join"""
    from string_splitter import StringSplitter
    from list_index_selector import ListIndexSelector
    
    splitter = StringSplitter()
    selector = ListIndexSelector()
    joiner = StringJoiner()
    
    # Start with frame numbers
    original = "10,25,42,100"
    
    # Split to list
    frame_list, _ = splitter.split_string(original, ",", output_type="INT")
    
    # Process - select one frame
    selected_frame, _ = selector.select_from_list(frame_list, 2, True)
    assert selected_frame == 42, "Should select frame 42"
    
    # Join back for display/logging
    display_string, _ = joiner.join_list(frame_list, ", ")
    assert display_string == "10, 25, 42, 100", f"Display string wrong: {display_string}"
    
    print("✓ test_full_workflow passed")


def run_all_tests():
    """Run all test functions"""
    print("Running tests for StringJoiner...\n")
    
    try:
        test_basic_join()
        test_with_string_splitter()
        test_integer_list()
        test_float_list()
        test_different_delimiters()
        test_edge_cases()
        test_mixed_types()
        test_return_types()
        test_input_types_structure()
        test_full_workflow()
        
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