"""
Tests for List Index Selector Node

Run with: python test_list_index_selector.py
"""

import inspect
from list_index_selector import ListIndexSelector


def test_basic_selection():
    """Test basic list item selection"""
    node = ListIndexSelector()
    
    test_list = ["a", "b", "c", "d"]
    
    result, length = node.select_from_list(test_list, 0)
    assert result == "a", f"Expected 'a', got {result}"
    assert length == 4, f"Expected length 4, got {length}"
    
    result, _ = node.select_from_list(test_list, 2)
    assert result == "c", f"Expected 'c', got {result}"
    
    print("✓ test_basic_selection passed")


def test_with_string_splitter_output():
    """Test using output from String Splitter"""
    # Simulate what String Splitter would output
    from string_splitter import StringSplitter
    
    splitter = StringSplitter()
    selector = ListIndexSelector()
    
    # Split a string
    string_list, _ = splitter.split_string("10,25,42,100", ",")
    
    # Select from the list
    result, _ = selector.select_from_list(string_list, 2)
    assert result == "42", f"Expected '42', got {result}"
    
    print("✓ test_with_string_splitter_output passed")


def test_indexing_modes():
    """Test zero vs one indexing"""
    node = ListIndexSelector()
    
    test_list = ["a", "b", "c"]
    
    # Zero-indexed
    result, _ = node.select_from_list(test_list, 0, zero_indexed=True)
    assert result == "a", f"Zero-indexed failed"
    
    # One-indexed
    result, _ = node.select_from_list(test_list, 1, zero_indexed=False)
    assert result == "a", f"One-indexed failed"
    
    result, _ = node.select_from_list(test_list, 2, zero_indexed=False)
    assert result == "b", f"One-indexed select 2 failed"
    
    print("✓ test_indexing_modes passed")


def test_different_types():
    """Test with different item types in list"""
    node = ListIndexSelector()
    
    # String list
    result, _ = node.select_from_list(["a", "b", "c"], 1)
    assert result == "b", f"String list failed"
    
    # Integer list
    result, _ = node.select_from_list([10, 20, 30], 0)
    assert result == 10, f"Integer list failed"
    
    # Mixed list
    result, _ = node.select_from_list(["text", 42, True], 1)
    assert result == 42, f"Mixed list failed"
    
    print("✓ test_different_types passed")


def test_out_of_range():
    """Test out of range handling"""
    node = ListIndexSelector()
    
    test_list = ["a", "b", "c"]
    
    # Index too high
    result, length = node.select_from_list(test_list, 10)
    assert result is None, f"Out of range should return None, got {result}"
    assert length == 3, f"Should still return correct length"
    
    # Negative with zero-indexing
    result, _ = node.select_from_list(test_list, -1)
    assert result is None, f"Negative index should return None"
    
    print("✓ test_out_of_range passed")


def test_edge_cases():
    """Test edge cases"""
    node = ListIndexSelector()
    
    # Empty list
    result, length = node.select_from_list([], 0)
    assert result is None, f"Empty list should return None"
    assert length == 0
    
    # Single item list
    result, length = node.select_from_list(["only"], 0)
    assert result == "only", f"Single item failed"
    assert length == 1
    
    print("✓ test_edge_cases passed")


def test_return_types():
    """Validate return types"""
    node = ListIndexSelector()
    
    result = node.select_from_list(["a", "b"], 0)
    
    assert isinstance(result, tuple), f"Should return tuple"
    assert len(result) == 2, f"Should return 2 items"
    # First item can be any type (*)
    assert isinstance(result[1], int), f"Second item should be int"
    
    print("✓ test_return_types passed")


def test_input_types_structure():
    """Validate INPUT_TYPES matches function signature"""
    input_types = ListIndexSelector.INPUT_TYPES()
    
    all_inputs = set()
    if "required" in input_types:
        all_inputs.update(input_types["required"].keys())
    if "optional" in input_types:
        all_inputs.update(input_types["optional"].keys())
    
    function = getattr(ListIndexSelector(), ListIndexSelector.FUNCTION)
    sig = inspect.signature(function)
    function_params = set(sig.parameters.keys()) - {'self'}
    
    missing = function_params - all_inputs
    extra = all_inputs - function_params
    
    assert not missing, f"Function has params not in INPUT_TYPES: {missing}"
    assert not extra, f"INPUT_TYPES has entries not in function: {extra}"
    
    print("✓ test_input_types_structure passed")


def run_all_tests():
    """Run all test functions"""
    print("Running tests for ListIndexSelector...\n")
    
    try:
        test_basic_selection()
        test_with_string_splitter_output()
        test_indexing_modes()
        test_different_types()
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
