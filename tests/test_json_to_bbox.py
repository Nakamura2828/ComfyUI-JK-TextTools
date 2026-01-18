"""
Tests for JSON to BBox Node

Run from project root:
    python tests/test_json_to_bbox.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import node modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import inspect
from json_to_bbox import JSONToBBox


def test_xyxy_to_xywh():
    """Test converting XYXY format to XYWH format"""
    node = JSONToBBox()

    # Two bboxes in XYXY format
    json_string = json.dumps([[10, 20, 110, 120], [200, 300, 250, 400]])

    bboxes, count = node.json_to_bbox(json_string, input_format="XYXY", output_format="XYWH")

    assert count == 2, f"Should have 2 bboxes, got {count}"
    assert len(bboxes) == 2, f"Should return 2 bboxes, got {len(bboxes)}"

    # First bbox: (10, 20, 110, 120) XYXY -> (10, 20, 100, 100) XYWH
    bbox1 = bboxes[0]
    assert isinstance(bbox1, list), "Each bbox should be wrapped in list"
    assert len(bbox1) == 1, "Each bbox should have one element (the coordinates)"
    coords1 = bbox1[0]
    assert coords1 == [10.0, 20.0, 100.0, 100.0], f"Expected [10, 20, 100, 100], got {coords1}"

    # Second bbox: (200, 300, 250, 400) XYXY -> (200, 300, 50, 100) XYWH
    coords2 = bboxes[1][0]
    assert coords2 == [200.0, 300.0, 50.0, 100.0], f"Expected [200, 300, 50, 100], got {coords2}"

    print("✓ test_xyxy_to_xywh passed")


def test_xywh_to_xyxy():
    """Test converting XYWH format to XYXY format"""
    node = JSONToBBox()

    # Two bboxes in XYWH format
    json_string = json.dumps([[10, 20, 100, 100], [200, 300, 50, 100]])

    bboxes, count = node.json_to_bbox(json_string, input_format="XYWH", output_format="XYXY")

    assert count == 2, f"Should have 2 bboxes, got {count}"

    # First bbox: (10, 20, 100, 100) XYWH -> (10, 20, 110, 120) XYXY
    coords1 = bboxes[0][0]
    assert coords1 == [10.0, 20.0, 110.0, 120.0], f"Expected [10, 20, 110, 120], got {coords1}"

    # Second bbox: (200, 300, 50, 100) XYWH -> (200, 300, 250, 400) XYXY
    coords2 = bboxes[1][0]
    assert coords2 == [200.0, 300.0, 250.0, 400.0], f"Expected [200, 300, 250, 400], got {coords2}"

    print("✓ test_xywh_to_xyxy passed")


def test_no_conversion_xyxy():
    """Test XYXY to XYXY (no conversion)"""
    node = JSONToBBox()

    json_string = json.dumps([[10, 20, 110, 120]])

    bboxes, count = node.json_to_bbox(json_string, input_format="XYXY", output_format="XYXY")

    assert count == 1, "Should have 1 bbox"
    coords = bboxes[0][0]
    assert coords == [10.0, 20.0, 110.0, 120.0], f"Coordinates should be unchanged, got {coords}"

    print("✓ test_no_conversion_xyxy passed")


def test_no_conversion_xywh():
    """Test XYWH to XYWH (no conversion)"""
    node = JSONToBBox()

    json_string = json.dumps([[10, 20, 100, 100]])

    bboxes, count = node.json_to_bbox(json_string, input_format="XYWH", output_format="XYWH")

    assert count == 1, "Should have 1 bbox"
    coords = bboxes[0][0]
    assert coords == [10.0, 20.0, 100.0, 100.0], f"Coordinates should be unchanged, got {coords}"

    print("✓ test_no_conversion_xywh passed")


def test_empty_array():
    """Test handling of empty JSON array"""
    node = JSONToBBox()

    json_string = "[]"

    bboxes, count = node.json_to_bbox(json_string, input_format="XYXY", output_format="XYWH")

    assert count == 0, "Should have 0 bboxes"
    assert bboxes == [], "Should return empty list"

    print("✓ test_empty_array passed")


def test_invalid_json():
    """Test handling of invalid JSON"""
    node = JSONToBBox()

    json_string = "not valid json {["

    bboxes, count = node.json_to_bbox(json_string, input_format="XYXY", output_format="XYWH")

    assert count == 0, "Should return 0 for invalid JSON"
    assert bboxes == [], "Should return empty list for invalid JSON"

    print("✓ test_invalid_json passed")


def test_non_array_json():
    """Test handling of JSON that's not an array"""
    node = JSONToBBox()

    # JSON object instead of array
    json_string = json.dumps({"bbox": [10, 20, 110, 120]})

    bboxes, count = node.json_to_bbox(json_string, input_format="XYXY", output_format="XYWH")

    assert count == 0, "Should return 0 for non-array JSON"
    assert bboxes == [], "Should return empty list for non-array JSON"

    print("✓ test_non_array_json passed")


def test_invalid_bbox_formats():
    """Test handling of invalid bbox formats in the array"""
    node = JSONToBBox()

    # Mix of valid and invalid bboxes
    json_string = json.dumps([
        [10, 20, 110, 120],      # Valid
        [30, 40],                 # Too few coordinates
        [50, 60, 70, 80, 90],    # Too many coordinates
        "not a bbox",             # Not a list
        [100, 200, 150, 250]     # Valid
    ])

    bboxes, count = node.json_to_bbox(json_string, input_format="XYXY", output_format="XYWH")

    assert count == 2, f"Should have 2 valid bboxes, got {count}"
    assert len(bboxes) == 2, "Should filter out invalid bboxes"

    # Check that only valid bboxes were processed
    coords1 = bboxes[0][0]
    assert coords1 == [10.0, 20.0, 100.0, 100.0], "First valid bbox"

    coords2 = bboxes[1][0]
    assert coords2 == [100.0, 200.0, 50.0, 50.0], "Second valid bbox"

    print("✓ test_invalid_bbox_formats passed")


def test_negative_coordinates():
    """Test handling of negative coordinates (valid in some contexts)"""
    node = JSONToBBox()

    # Bbox with negative coordinates
    json_string = json.dumps([[-10, -20, 50, 60]])

    bboxes, count = node.json_to_bbox(json_string, input_format="XYXY", output_format="XYWH")

    assert count == 1, "Should handle negative coordinates"
    coords = bboxes[0][0]
    # (-10, -20, 50, 60) XYXY -> (-10, -20, 60, 80) XYWH
    assert coords == [-10.0, -20.0, 60.0, 80.0], f"Should handle negative coords, got {coords}"

    print("✓ test_negative_coordinates passed")


def test_zero_size_bbox():
    """Test handling of zero-width or zero-height bbox"""
    node = JSONToBBox()

    # Zero-width bbox in XYXY: x1 == x2
    json_string = json.dumps([[100, 100, 100, 200]])

    bboxes, count = node.json_to_bbox(json_string, input_format="XYXY", output_format="XYWH")

    assert count == 1, "Should process zero-width bbox"
    coords = bboxes[0][0]
    # (100, 100, 100, 200) XYXY -> (100, 100, 0, 100) XYWH
    assert coords == [100.0, 100.0, 0.0, 100.0], f"Should handle zero width, got {coords}"

    print("✓ test_zero_size_bbox passed")


def test_float_coordinates():
    """Test handling of floating point coordinates"""
    node = JSONToBBox()

    json_string = json.dumps([[10.5, 20.7, 110.3, 120.9]])

    bboxes, count = node.json_to_bbox(json_string, input_format="XYXY", output_format="XYWH")

    assert count == 1, "Should handle float coordinates"
    coords = bboxes[0][0]
    # (10.5, 20.7, 110.3, 120.9) XYXY -> (10.5, 20.7, 99.8, 100.2) XYWH
    expected = [10.5, 20.7, 99.8, 100.19999999999999]  # Floating point precision
    assert coords[0] == expected[0], "X coordinate"
    assert coords[1] == expected[1], "Y coordinate"
    assert abs(coords[2] - 99.8) < 0.001, "Width should be approximately 99.8"
    assert abs(coords[3] - 100.2) < 0.001, "Height should be approximately 100.2"

    print("✓ test_float_coordinates passed")


def test_large_number_of_bboxes():
    """Test handling many bboxes"""
    node = JSONToBBox()

    # Create 100 bboxes
    bboxes_input = [[i*10, i*10, i*10+50, i*10+50] for i in range(100)]
    json_string = json.dumps(bboxes_input)

    bboxes, count = node.json_to_bbox(json_string, input_format="XYXY", output_format="XYWH")

    assert count == 100, f"Should have 100 bboxes, got {count}"
    assert len(bboxes) == 100, "Should return 100 bboxes"

    # Check first and last
    assert bboxes[0][0] == [0.0, 0.0, 50.0, 50.0], "First bbox"
    assert bboxes[99][0] == [990.0, 990.0, 50.0, 50.0], "Last bbox"

    print("✓ test_large_number_of_bboxes passed")


def test_output_format():
    """Test that output format matches BBOX type requirements"""
    node = JSONToBBox()

    json_string = json.dumps([[10, 20, 110, 120], [200, 300, 250, 400]])

    bboxes, count = node.json_to_bbox(json_string, input_format="XYXY", output_format="XYWH")

    # Verify structure: [[[x,y,w,h]], [[x,y,w,h]], ...]
    assert isinstance(bboxes, list), "bboxes should be a list"
    assert len(bboxes) == 2, "Should have 2 elements"

    for i, bbox in enumerate(bboxes):
        assert isinstance(bbox, list), f"bbox {i} should be a list"
        assert len(bbox) == 1, f"bbox {i} should have one element (wrapped)"
        assert isinstance(bbox[0], list), f"bbox {i} inner should be a list"
        assert len(bbox[0]) == 4, f"bbox {i} should have 4 coordinates"
        for coord in bbox[0]:
            assert isinstance(coord, float), f"All coordinates should be floats"

    print("✓ test_output_format passed")


def test_return_types():
    """Validate return types match OUTPUT_IS_LIST specification"""
    node = JSONToBBox()

    json_string = json.dumps([[10, 20, 110, 120]])

    result = node.json_to_bbox(json_string, input_format="XYXY", output_format="XYWH")

    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 2, "Should return 2 items (bboxes, count)"

    bboxes, count = result

    assert isinstance(bboxes, list), "bboxes should be list (for OUTPUT_IS_LIST)"
    assert isinstance(count, int), "count should be int"

    print("✓ test_return_types passed")


def test_input_types_structure():
    """Validate INPUT_TYPES matches function signature"""
    input_types = JSONToBBox.INPUT_TYPES()

    all_inputs = set()
    if "required" in input_types:
        all_inputs.update(input_types["required"].keys())
    if "optional" in input_types:
        all_inputs.update(input_types["optional"].keys())

    function = getattr(JSONToBBox(), JSONToBBox.FUNCTION)
    sig = inspect.signature(function)
    function_params = set(sig.parameters.keys()) - {'self'}

    missing = function_params - all_inputs
    extra = all_inputs - function_params

    assert not missing, f"Function has params not in INPUT_TYPES: {missing}"
    assert not extra, f"INPUT_TYPES has entries not in function: {extra}"

    print("✓ test_input_types_structure passed")


def test_sam3_format():
    """Test typical SAM3 Segmentation output format"""
    node = JSONToBBox()

    # Typical SAM3 boxes output (XYXY format)
    sam3_json = json.dumps([
        [245.3, 167.8, 512.6, 389.2],
        [100.0, 200.0, 300.0, 400.0],
        [450.5, 100.3, 600.7, 250.9]
    ])

    bboxes, count = node.json_to_bbox(sam3_json, input_format="XYXY", output_format="XYWH")

    assert count == 3, f"Should extract 3 bboxes from SAM3 output, got {count}"

    # Verify first bbox conversion
    coords = bboxes[0][0]
    expected_w = 512.6 - 245.3
    expected_h = 389.2 - 167.8
    assert abs(coords[0] - 245.3) < 0.001, "X coordinate"
    assert abs(coords[1] - 167.8) < 0.001, "Y coordinate"
    assert abs(coords[2] - expected_w) < 0.001, "Width"
    assert abs(coords[3] - expected_h) < 0.001, "Height"

    print("✓ test_sam3_format passed")


def run_all_tests():
    """Run all test functions"""
    print("Running tests for JSONToBBox...\n")

    try:
        test_xyxy_to_xywh()
        test_xywh_to_xyxy()
        test_no_conversion_xyxy()
        test_no_conversion_xywh()
        test_empty_array()
        test_invalid_json()
        test_non_array_json()
        test_invalid_bbox_formats()
        test_negative_coordinates()
        test_zero_size_bbox()
        test_float_coordinates()
        test_large_number_of_bboxes()
        test_output_format()
        test_return_types()
        test_input_types_structure()
        test_sam3_format()

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
