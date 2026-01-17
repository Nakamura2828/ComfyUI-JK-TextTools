"""
Tests for JSON Pretty Printer Node

Run with: python test_json_pretty_printer.py
"""

import json
import inspect
from json_pretty_printer import JSONPrettyPrinter


def test_basic_formatting():
    """Test basic JSON formatting"""
    node = JSONPrettyPrinter()
    
    # Compact JSON
    input_json = '{"name":"Alice","age":30}'
    formatted, is_valid, error = node.format_json(input_json)
    
    assert is_valid == True, "Valid JSON should be marked as valid"
    assert error == "", "No error for valid JSON"
    assert "Alice" in formatted, "Content should be preserved"
    assert "\n" in formatted, "Should have newlines for readability"
    
    print("✓ test_basic_formatting passed")


def test_detection_data():
    """Test formatting your actual detection data"""
    node = JSONPrettyPrinter()
    
    # Your detection data (simplified)
    input_json = '[{"detect_result":[{"class":"CLASS1_LABEL","score":0.838,"box":[246,149,174,207]}],"categorization":false}]'
    
    formatted, is_valid, error = node.format_json(input_json)
    
    assert is_valid == True, "Detection data should be valid"
    assert "detect_result" in formatted
    assert "CLASS1_LABEL" in formatted
    # Should be more readable than input
    assert len(formatted) > len(input_json), "Formatted should be longer (more whitespace)"
    
    print("✓ test_detection_data passed")


def test_indentation_levels():
    """Test different indentation settings"""
    node = JSONPrettyPrinter()
    
    input_json = '{"a":{"b":{"c":1}}}'
    
    # 2-space indent (default)
    formatted_2, _, _ = node.format_json(input_json, indent=2)
    assert "  " in formatted_2, "Should have 2-space indentation"
    
    # 4-space indent
    formatted_4, _, _ = node.format_json(input_json, indent=4)
    assert "    " in formatted_4, "Should have 4-space indentation"
    
    # No indent (compact-ish, but still has newlines)
    formatted_0, _, _ = node.format_json(input_json, indent=0)
    
    print("✓ test_indentation_levels passed")


def test_sort_keys():
    """Test key sorting option"""
    node = JSONPrettyPrinter()
    
    input_json = '{"z":1,"a":2,"m":3}'
    
    # Without sorting
    formatted_unsorted, _, _ = node.format_json(input_json, sort_keys=False)
    
    # With sorting
    formatted_sorted, _, _ = node.format_json(input_json, sort_keys=True)
    
    # Parse both to check order
    unsorted_obj = json.loads(formatted_unsorted)
    sorted_obj = json.loads(formatted_sorted)
    
    # Should have same content
    assert unsorted_obj == sorted_obj
    
    # Check if sorted version has keys in order
    # (This is a bit tricky since dict order is preserved in Python 3.7+)
    sorted_keys = list(sorted_obj.keys())
    assert sorted_keys == ["a", "m", "z"], "Keys should be alphabetically sorted"
    
    print("✓ test_sort_keys passed")


def test_compact_mode():
    """Test compact output mode"""
    node = JSONPrettyPrinter()
    
    input_json = '{"name": "Alice", "age": 30}'
    
    # Normal mode
    formatted_normal, _, _ = node.format_json(input_json, indent=2, compact=False)
    
    # Compact mode
    formatted_compact, _, _ = node.format_json(input_json, compact=True)
    
    # Compact should have no extra whitespace
    assert "\n" not in formatted_compact, "Compact mode should have no newlines"
    assert formatted_compact == '{"name":"Alice","age":30}', "Should be truly compact"
    
    # Normal should be longer
    assert len(formatted_normal) > len(formatted_compact)
    
    print("✓ test_compact_mode passed")


def test_invalid_json():
    """Test handling of invalid JSON"""
    node = JSONPrettyPrinter()
    
    # Missing closing brace
    invalid_json = '{"name":"Alice"'
    formatted, is_valid, error = node.format_json(invalid_json)
    
    assert is_valid == False, "Should detect invalid JSON"
    assert len(error) > 0, "Should provide error message"
    assert "line" in error.lower() or "column" in error.lower(), "Error should mention location"
    assert formatted == invalid_json, "Should return original on error"
    
    print("✓ test_invalid_json passed")


def test_various_json_types():
    """Test with different JSON structures"""
    node = JSONPrettyPrinter()
    
    # Array
    array_json = '[1,2,3,4,5]'
    formatted, is_valid, _ = node.format_json(array_json)
    assert is_valid == True
    
    # Nested object
    nested_json = '{"a":{"b":{"c":{"d":1}}}}'
    formatted, is_valid, _ = node.format_json(nested_json)
    assert is_valid == True
    
    # Mixed
    mixed_json = '{"items":[{"id":1},{"id":2}],"count":2}'
    formatted, is_valid, _ = node.format_json(mixed_json)
    assert is_valid == True
    
    # Empty
    empty_obj = '{}'
    formatted, is_valid, _ = node.format_json(empty_obj)
    assert is_valid == True
    
    empty_array = '[]'
    formatted, is_valid, _ = node.format_json(empty_array)
    assert is_valid == True
    
    print("✓ test_various_json_types passed")


def test_special_characters():
    """Test JSON with special characters"""
    node = JSONPrettyPrinter()
    
    # Quotes, newlines, unicode
    special_json = '{"text":"He said \\"Hello\\"","newline":"Line1\\nLine2","unicode":"こんにちは"}'
    formatted, is_valid, error = node.format_json(special_json)
    
    assert is_valid == True, f"Should handle special chars, error: {error}"
    assert "Hello" in formatted
    
    print("✓ test_special_characters passed")


def test_return_types():
    """Validate return types"""
    node = JSONPrettyPrinter()
    
    result = node.format_json('{"test":true}')
    
    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 3, "Should return 3 items"
    assert isinstance(result[0], str), "First item should be string"
    assert isinstance(result[1], bool), "Second item should be boolean"
    assert isinstance(result[2], str), "Third item should be string"
    
    print("✓ test_return_types passed")


def test_input_types_structure():
    """Validate INPUT_TYPES matches function signature"""
    input_types = JSONPrettyPrinter.INPUT_TYPES()
    
    all_inputs = set()
    if "required" in input_types:
        all_inputs.update(input_types["required"].keys())
    if "optional" in input_types:
        all_inputs.update(input_types["optional"].keys())
    
    function = getattr(JSONPrettyPrinter(), JSONPrettyPrinter.FUNCTION)
    sig = inspect.signature(function)
    function_params = set(sig.parameters.keys()) - {'self'}
    
    missing = function_params - all_inputs
    extra = all_inputs - function_params
    
    assert not missing, f"Function has params not in INPUT_TYPES: {missing}"
    assert not extra, f"INPUT_TYPES has entries not in function: {extra}"
    
    print("✓ test_input_types_structure passed")


def run_all_tests():
    """Run all test functions"""
    print("Running tests for JSONPrettyPrinter...\n")
    
    try:
        test_basic_formatting()
        test_detection_data()
        test_indentation_levels()
        test_sort_keys()
        test_compact_mode()
        test_invalid_json()
        test_various_json_types()
        test_special_characters()
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