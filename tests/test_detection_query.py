"""
Tests for Detection Query Node

Run with: python test_detection_query.py
"""

import json
import inspect
import sys
from pathlib import Path

# Add parent directory to path to import node modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection_query import DetectionQuery


# Sample detection data for testing
SAMPLE_DETECTIONS = [
    {
        "detect_result": [
            {"class": "CLASS1_LABEL", "score": 0.838, "box": [246, 149, 174, 207]},
            {"class": "CLASS2_LABEL", "score": 0.776, "box": [191, 506, 204, 189]},
            {"class": "CLASS2_LABEL", "score": 0.770, "box": [366, 532, 226, 195]},
            {"class": "CLASS3_LABEL", "score": 0.658, "box": [557, 527, 67, 86]},
            {"class": "CLASS1_SUBCLASS1", "score": 0.521, "box": [210, 735, 301, 183]},
            {"class": "CLASS1_SUBCLASS2", "score": 0.494, "box": [141, 496, 100, 103]}
        ],
        "categorization": False
    }
]


def test_all_detections():
    """Test getting all detections with wildcard"""
    node = DetectionQuery()
    
    input_json = json.dumps(SAMPLE_DETECTIONS)
    filtered_json, count, detection_list, _, is_valid, error = node.query_detections(
        input_json, 
        class_filter="*"
    )
    
    assert is_valid == True, "Should be valid"
    assert count == 6, f"Should return all 6 detections, got {count}"
    assert len(detection_list) == 6, "List should have 6 items"
    
    print("✓ test_all_detections passed")


def test_exact_class_match():
    """Test exact class name matching"""
    node = DetectionQuery()
    
    input_json = json.dumps(SAMPLE_DETECTIONS)
    filtered_json, count, detection_list, _, is_valid, error = node.query_detections(
        input_json,
        class_filter="CLASS1_LABEL"
    )
    
    assert is_valid == True
    assert count == 1, f"Should find 1 CLASS1_LABEL, found {count}"
    assert detection_list[0]["class"] == "CLASS1_LABEL"
    assert detection_list[0]["score"] == 0.838
    
    print("✓ test_exact_class_match passed")


def test_wildcard_prefix():
    """Test wildcard prefix matching"""
    node = DetectionQuery()
    
    input_json = json.dumps(SAMPLE_DETECTIONS)
    
    # Match all CLASS1_* variants
    filtered_json, count, detection_list, _, is_valid, error = node.query_detections(
        input_json,
        class_filter="CLASS1_*"
    )
    
    assert count == 3, f"Should find 3 CLASS1_* items, found {count}"
    
    # Verify all start with CLASS1_
    for detection in detection_list:
        assert detection["class"].startswith("CLASS1_"), \
            f"All should start with CLASS1_, got {detection['class']}"
    
    print("✓ test_wildcard_prefix passed")


def test_wildcard_suffix():
    """Test wildcard suffix matching"""
    node = DetectionQuery()
    
    input_json = json.dumps(SAMPLE_DETECTIONS)
    
    # Match all *_LABEL classes
    filtered_json, count, detection_list, _, is_valid, error = node.query_detections(
        input_json,
        class_filter="*_LABEL"
    )
    
    assert count == 4, f"Should find 4 *_LABEL items, found {count}"
    
    # Verify all end with _LABEL
    for detection in detection_list:
        assert detection["class"].endswith("_LABEL"), \
            f"All should end with _LABEL, got {detection['class']}"
    
    print("✓ test_wildcard_suffix passed")


def test_score_filtering():
    """Test minimum score filtering"""
    node = DetectionQuery()
    
    input_json = json.dumps(SAMPLE_DETECTIONS)
    
    # Only high confidence (> 0.7)
    filtered_json, count, detection_list, _, is_valid, error = node.query_detections(
        input_json,
        class_filter="*",
        min_score=0.7
    )
    
    assert count == 3, f"Should find 3 high-confidence detections, found {count}"
    
    # Verify all scores >= 0.7
    for detection in detection_list:
        assert detection["score"] >= 0.7, \
            f"All scores should be >= 0.7, got {detection['score']}"
    
    print("✓ test_score_filtering passed")


def test_combined_filters():
    """Test combining class and score filters"""
    node = DetectionQuery()
    
    input_json = json.dumps(SAMPLE_DETECTIONS)
    
    # CLASS2_* with score > 0.77
    filtered_json, count, detection_list, _, is_valid, error = node.query_detections(
        input_json,
        class_filter="CLASS2_*",
        min_score=0.77
    )
    
    assert count == 2, f"Should find 2 matching detections, found {count}"
    
    # Verify filters
    for detection in detection_list:
        assert detection["class"].startswith("CLASS2_")
        assert detection["score"] >= 0.77
    
    print("✓ test_combined_filters passed")


def test_max_results():
    """Test limiting number of results"""
    node = DetectionQuery()
    
    input_json = json.dumps(SAMPLE_DETECTIONS)
    
    # Get only first 2 results
    filtered_json, count, detection_list, _, is_valid, error = node.query_detections(
        input_json,
        class_filter="*",
        max_results=2
    )
    
    assert count == 2, f"Should limit to 2 results, got {count}"
    assert len(detection_list) == 2
    
    print("✓ test_max_results passed")


def test_no_matches():
    """Test when no detections match filters"""
    node = DetectionQuery()
    
    input_json = json.dumps(SAMPLE_DETECTIONS)
    
    # Non-existent class
    filtered_json, count, detection_list, _, is_valid, error = node.query_detections(
        input_json,
        class_filter="NONEXISTENT_CLASS"
    )
    
    assert is_valid == True, "Should still be valid even with no matches"
    assert count == 0, "Should find 0 matches"
    assert len(detection_list) == 0, "List should be empty"
    
    print("✓ test_no_matches passed")


def test_json_output_format():
    """Test that JSON output is properly formatted"""
    node = DetectionQuery()
    
    input_json = json.dumps(SAMPLE_DETECTIONS)
    filtered_json, count, detection_list, _, is_valid, error = node.query_detections(
        input_json,
        class_filter="CLASS1_LABEL"
    )
    
    # Parse the output JSON
    output_data = json.loads(filtered_json)
    
    # Should maintain wrapper format
    assert isinstance(output_data, list)
    assert "detect_result" in output_data[0]
    assert len(output_data[0]["detect_result"]) == 1
    
    print("✓ test_json_output_format passed")


def test_detection_list_output():
    """Test that detection_list is proper list for OUTPUT_IS_LIST"""
    node = DetectionQuery()
    
    input_json = json.dumps(SAMPLE_DETECTIONS)
    filtered_json, count, detection_list, _, is_valid, error = node.query_detections(
        input_json,
        class_filter="CLASS2_*"
    )
    
    # Should be a Python list
    assert isinstance(detection_list, list)
    assert len(detection_list) == 2
    
    # Each item should be a dict with detection data
    for detection in detection_list:
        assert isinstance(detection, dict)
        assert "class" in detection
        assert "score" in detection
        assert "box" in detection
    
    print("✓ test_detection_list_output passed")


def test_invalid_json():
    """Test handling of invalid JSON"""
    node = DetectionQuery()
    
    invalid_json = '{"broken": invalid}'
    filtered_json, count, detection_list, _, is_valid, error = node.query_detections(
        invalid_json,
        class_filter="*"
    )
    
    assert is_valid == False, "Should detect invalid JSON"
    assert count == 0, "Count should be 0 on error"
    assert len(error) > 0, "Should provide error message"
    
    print("✓ test_invalid_json passed")


def test_simple_list_format():
    """Test with simple list format (no wrapper)"""
    node = DetectionQuery()
    
    # Simple list without wrapper
    simple_data = [
        {"class": "CLASS1", "score": 0.9, "box": [1, 2, 3, 4]},
        {"class": "CLASS2", "score": 0.8, "box": [5, 6, 7, 8]}
    ]
    
    input_json = json.dumps(simple_data)
    filtered_json, count, detection_list, _, is_valid, error = node.query_detections(
        input_json,
        class_filter="CLASS1"
    )
    
    assert is_valid == True
    assert count == 1
    assert detection_list[0]["class"] == "CLASS1"
    
    print("✓ test_simple_list_format passed")


def test_categorization_extraction():
    """Test extracting categorization field"""
    node = DetectionQuery()
    
    # Data with is_dog field
    test_data = [{
        "detect_result": [
            {"class": "DOG", "score": 0.9, "box": [1, 2, 3, 4]}
        ],
        "is_dog": True
    }]
    
    input_json = json.dumps(test_data)
    _, _, _, cat_value, is_valid, _ = node.query_detections(
        input_json,
        class_filter="*",
        categorization_field="is_dog"
    )
    
    assert is_valid == True
    assert cat_value == True, f"Should extract is_dog=True, got {cat_value}"
    
    # Test with False value
    test_data[0]["is_dog"] = False
    input_json = json.dumps(test_data)
    _, _, _, cat_value, _, _ = node.query_detections(
        input_json,
        class_filter="*",
        categorization_field="is_dog"
    )
    
    assert cat_value == False, f"Should extract is_dog=False, got {cat_value}"
    
    print("✓ test_categorization_extraction passed")


def test_categorization_field_not_found():
    """Test when categorization field doesn't exist"""
    node = DetectionQuery()
    
    input_json = json.dumps(SAMPLE_DETECTIONS)
    _, _, _, cat_value, is_valid, _ = node.query_detections(
        input_json,
        class_filter="*",
        categorization_field="nonexistent_field"
    )
    
    assert is_valid == True
    assert cat_value is None, f"Non-existent field should return None, got {cat_value}"
    
    print("✓ test_categorization_field_not_found passed")


def test_categorization_with_different_types():
    """Test categorization field with different value types"""
    node = DetectionQuery()
    
    # String value
    test_data = [{
        "detect_result": [{"class": "A", "score": 0.9, "box": [1, 2, 3, 4]}],
        "category": "animal"
    }]
    input_json = json.dumps(test_data)
    _, _, _, cat_value, _, _ = node.query_detections(
        input_json, "*", categorization_field="category"
    )
    assert cat_value == "animal"
    
    # Number value
    test_data[0]["count"] = 42
    input_json = json.dumps(test_data)
    _, _, _, cat_value, _, _ = node.query_detections(
        input_json, "*", categorization_field="count"
    )
    assert cat_value == 42
    
    print("✓ test_categorization_with_different_types passed")


def test_return_types():
    """Validate return types"""
    node = DetectionQuery()
    
    result = node.query_detections(json.dumps(SAMPLE_DETECTIONS), "*")
    
    assert isinstance(result, tuple)
    assert len(result) == 6, f"Should return 6 items, got {len(result)}"
    assert isinstance(result[0], str), "filtered_json should be string"
    assert isinstance(result[1], int), "match_count should be int"
    assert isinstance(result[2], list), "detection_list should be list"
    # result[3] is any_typ (categorization_value) - can be any type
    assert isinstance(result[4], bool), "is_valid should be bool"
    assert isinstance(result[5], str), "error_message should be string"
    
    print("✓ test_return_types passed")


def test_input_types_structure():
    """Validate INPUT_TYPES matches function signature"""
    input_types = DetectionQuery.INPUT_TYPES()
    
    all_inputs = set()
    if "required" in input_types:
        all_inputs.update(input_types["required"].keys())
    if "optional" in input_types:
        all_inputs.update(input_types["optional"].keys())
    
    function = getattr(DetectionQuery(), DetectionQuery.FUNCTION)
    sig = inspect.signature(function)
    function_params = set(sig.parameters.keys()) - {'self'}
    
    missing = function_params - all_inputs
    extra = all_inputs - function_params
    
    assert not missing, f"Function has params not in INPUT_TYPES: {missing}"
    assert not extra, f"INPUT_TYPES has entries not in function: {extra}"
    
    print("✓ test_input_types_structure passed")


def run_all_tests():
    """Run all test functions"""
    print("Running tests for DetectionQuery...\n")
    
    try:
        test_all_detections()
        test_exact_class_match()
        test_wildcard_prefix()
        test_wildcard_suffix()
        test_score_filtering()
        test_combined_filters()
        test_max_results()
        test_no_matches()
        test_json_output_format()
        test_detection_list_output()
        test_invalid_json()
        test_simple_list_format()
        test_categorization_extraction()
        test_categorization_field_not_found()
        test_categorization_with_different_types()
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