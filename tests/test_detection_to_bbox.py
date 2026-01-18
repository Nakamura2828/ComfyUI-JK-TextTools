"""
Tests for Detection to BBox Node

Run from project root:
    python tests/test_detection_to_bbox.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import node modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import inspect
from detection_to_bbox import DetectionToBBox


def test_basic_extraction():
    """Test basic bbox extraction"""
    node = DetectionToBBox()
    
    detection = {
        "class": "DOG",
        "score": 0.95,
        "box": [100, 200, 50, 75]
    }
    
    bbox, x, y, w, h, class_name, score = node.extract_bbox(json.dumps(detection))
    
    # bbox is now a tensor
    assert isinstance(bbox, list), f"Expected list, got {type(bbox)}"
    assert len(bbox) == 1 and len(bbox[0])==4, f"Expected  list of 1 list of 4 elements, got {len(bbox)} and {len(bbox[0])}"
    assert bbox == [[100, 200, 50, 75]], f"Expected [[100, 200, 50, 75]], got {bbox}"
    
    assert x == 100, f"Expected x=100, got {x}"
    assert y == 200, f"Expected y=200, got {y}"
    assert w == 50, f"Expected w=50, got {w}"
    assert h == 75, f"Expected h=75, got {h}"
    assert class_name == "DOG", f"Expected class 'DOG', got '{class_name}'"
    assert score == 0.95, f"Expected score 0.95, got {score}"
    
    print("✓ test_basic_extraction passed")


def test_bbox_key_variations():
    """Test different bbox key names"""
    node = DetectionToBBox()
    
    # Using "box"
    detection = {"box": [10, 20, 30, 40], "class": "CAT", "score": 0.8}
    bbox, x, y, w, h, _, _ = node.extract_bbox(json.dumps(detection), bbox_key="box")
    assert bbox == [[10, 20, 30, 40]]
    
    # Using "bbox"
    detection = {"bbox": [15, 25, 35, 45], "class": "CAT", "score": 0.8}
    bbox, x, y, w, h, _, _ = node.extract_bbox(json.dumps(detection), bbox_key="bbox")
    assert bbox == [[15, 25, 35, 45]]
    
    # Auto-detect "box"
    detection = {"box": [5, 6, 7, 8], "class": "BIRD", "score": 0.7}
    bbox, _, _, _, _, _, _ = node.extract_bbox(json.dumps(detection))
    assert bbox == [[5, 6, 7, 8]]
    
    # Auto-detect "bbox"
    detection = {"bbox": [1, 2, 3, 4], "class": "FISH", "score": 0.6}
    bbox, _, _, _, _, _, _ = node.extract_bbox(json.dumps(detection))
    assert bbox == [[1, 2, 3, 4]]
    
    print("✓ test_bbox_key_variations passed")


def test_from_detection_query():
    """Test with actual Detection Query output format"""
    node = DetectionToBBox()
    
    # This is what Detection Query outputs in detection_list
    detection = {
        "class": "CLASS1_LABEL",
        "score": 0.838,
        "box": [246, 149, 174, 207]
    }
    
    bbox, x, y, w, h, class_name, score = node.extract_bbox(json.dumps(detection))
    
    assert bbox == [[246, 149, 174, 207]]
    assert x == 246
    assert y == 149
    assert w == 174
    assert h == 207
    assert class_name == "CLASS1_LABEL"
    assert score == 0.838
    
    print("✓ test_from_detection_query passed")


def test_integer_conversion():
    """Test that bbox values are converted to integers"""
    node = DetectionToBBox()
    
    # Float values in bbox
    detection = {
        "box": [10.7, 20.3, 30.9, 40.1],
        "class": "TEST",
        "score": 0.5
    }
    
    bbox, x, y, w, h, _, _ = node.extract_bbox(json.dumps(detection))
    
    # Check types
    assert isinstance(x, int), f"x should be int, got {type(x)}"
    assert isinstance(y, int), f"y should be int, got {type(y)}"
    assert isinstance(w, int), f"w should be int, got {type(w)}"
    assert isinstance(h, int), f"h should be int, got {type(h)}"
    
    # Check values (should be truncated)
    assert x == 10
    assert y == 20
    assert w == 30
    assert h == 40
    
    print("✓ test_integer_conversion passed")


def test_missing_bbox():
    """Test handling when bbox is missing"""
    node = DetectionToBBox()
    
    # No bbox at all
    detection = {"class": "NO_BOX", "score": 0.5}
    
    bbox, x, y, w, h, class_name, score = node.extract_bbox(json.dumps(detection))
    
    assert bbox == [[0, 0, 0, 0]], "Should return zeros for missing bbox"
    assert x == 0 and y == 0 and w == 0 and h == 0
    assert class_name == "NO_BOX"  # Should still extract class
    
    print("✓ test_missing_bbox passed")


def test_invalid_bbox_length():
    """Test handling of malformed bbox"""
    node = DetectionToBBox()
    
    # Too few values
    detection = {"box": [10, 20], "class": "BAD", "score": 0.5}
    bbox, _, _, _, _, _, _ = node.extract_bbox(json.dumps(detection))
    assert bbox == [[0, 0, 0, 0]], "Should return zeros for invalid bbox"
    
    # Too many values
    detection = {"box": [10, 20, 30, 40, 50], "class": "BAD", "score": 0.5}
    bbox, _, _, _, _, _, _ = node.extract_bbox(json.dumps(detection))
    assert bbox == [[0, 0, 0, 0]], "Should return zeros for invalid bbox"
    
    print("✓ test_invalid_bbox_length passed")


def test_missing_optional_fields():
    """Test when class or score are missing"""
    node = DetectionToBBox()
    
    # No class or score
    detection = {"box": [10, 20, 30, 40]}
    
    bbox, x, y, w, h, class_name, score = node.extract_bbox(json.dumps(detection))
    
    assert bbox == [[10, 20, 30, 40]], "Should still extract bbox"
    assert class_name == "", "Missing class should return empty string"
    assert score == 0.0, "Missing score should return 0.0"
    
    print("✓ test_missing_optional_fields passed")


def test_return_types():
    """Validate return types"""
    node = DetectionToBBox()
    
    detection = {"box": [10, 20, 30, 40], "class": "TEST", "score": 0.9}
    result = node.extract_bbox(json.dumps(detection))
    
    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 7, f"Should return 7 items, got {len(result)}"
    
    bbox, x, y, w, h, class_name, score = result
    
    assert isinstance(bbox, list), "bbox should be list"
    assert isinstance(x, int), "x should be int"
    assert isinstance(y, int), "y should be int"
    assert isinstance(w, int), "w should be int"
    assert isinstance(h, int), "h should be int"
    assert isinstance(class_name, str), "class_name should be str"
    assert isinstance(score, float), "score should be float"
    
    print("✓ test_return_types passed")


def test_input_types_structure():
    """Validate INPUT_TYPES matches function signature"""
    input_types = DetectionToBBox.INPUT_TYPES()
    
    all_inputs = set()
    if "required" in input_types:
        all_inputs.update(input_types["required"].keys())
    if "optional" in input_types:
        all_inputs.update(input_types["optional"].keys())
    
    function = getattr(DetectionToBBox(), DetectionToBBox.FUNCTION)
    sig = inspect.signature(function)
    function_params = set(sig.parameters.keys()) - {'self'}
    
    missing = function_params - all_inputs
    extra = all_inputs - function_params
    
    assert not missing, f"Function has params not in INPUT_TYPES: {missing}"
    assert not extra, f"INPUT_TYPES has entries not in function: {extra}"
    
    print("✓ test_input_types_structure passed")


def run_all_tests():
    """Run all test functions"""
    print("Running tests for DetectionToBBox...\n")
    
    try:
        test_basic_extraction()
        test_bbox_key_variations()
        test_from_detection_query()
        test_integer_conversion()
        test_missing_bbox()
        test_invalid_bbox_length()
        test_missing_optional_fields()
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
