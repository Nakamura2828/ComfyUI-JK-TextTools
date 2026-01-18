"""
Tests for BBox to SAM3 Query Node

Test the conversion of BBOX format to SAM3 Selector query formats.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bbox_to_sam3_query import BBoxToSAM3Query


def test_basic_conversion():
    """Test basic XYWH to XYXY conversion for box query"""
    node = BBoxToSAM3Query()

    # BBOX: x=100, y=200, w=50, h=75
    bbox = [[100, 200, 50, 75]]

    box_query_str, point_query_str = node.bbox_to_sam3_query(bbox)

    # Parse JSON strings
    box_query = json.loads(box_query_str)
    point_query = json.loads(point_query_str)

    # Check box query (XYXY format)
    assert len(box_query) == 1, "Should have one box"
    assert box_query[0]["x1"] == 100.0, "x1 should be x"
    assert box_query[0]["y1"] == 200.0, "y1 should be y"
    assert box_query[0]["x2"] == 150.0, "x2 should be x + w"
    assert box_query[0]["y2"] == 275.0, "y2 should be y + h"

    # Check point query (center point)
    assert len(point_query) == 1, "Should have one point"
    assert point_query[0]["x"] == 125.0, "center_x should be x + w/2"
    assert point_query[0]["y"] == 237.5, "center_y should be y + h/2"

    print("✓ test_basic_conversion passed")


def test_unwrapped_bbox_format():
    """Test handling of unwrapped bbox format [x,y,w,h]"""
    node = BBoxToSAM3Query()

    # Unwrapped format (not nested)
    bbox = [50, 60, 100, 120]

    box_query_str, point_query_str = node.bbox_to_sam3_query(bbox)

    box_query = json.loads(box_query_str)
    point_query = json.loads(point_query_str)

    # Check conversions
    assert box_query[0]["x1"] == 50.0
    assert box_query[0]["y1"] == 60.0
    assert box_query[0]["x2"] == 150.0  # 50 + 100
    assert box_query[0]["y2"] == 180.0  # 60 + 120

    assert point_query[0]["x"] == 100.0  # 50 + 100/2
    assert point_query[0]["y"] == 120.0  # 60 + 120/2

    print("✓ test_unwrapped_bbox_format passed")


def test_integer_coordinates():
    """Test that integer coordinates are converted to floats"""
    node = BBoxToSAM3Query()

    # Pure integer coordinates
    bbox = [[10, 20, 30, 40]]

    box_query_str, point_query_str = node.bbox_to_sam3_query(bbox)

    box_query = json.loads(box_query_str)
    point_query = json.loads(point_query_str)

    # Verify all values are floats in the output
    assert isinstance(box_query[0]["x1"], float)
    assert isinstance(box_query[0]["y1"], float)
    assert isinstance(box_query[0]["x2"], float)
    assert isinstance(box_query[0]["y2"], float)

    assert isinstance(point_query[0]["x"], float)
    assert isinstance(point_query[0]["y"], float)

    print("✓ test_integer_coordinates passed")


def test_zero_size_bbox():
    """Test handling of zero-width or zero-height bbox"""
    node = BBoxToSAM3Query()

    # Zero width
    bbox = [[100, 200, 0, 50]]
    box_query_str, point_query_str = node.bbox_to_sam3_query(bbox)

    box_query = json.loads(box_query_str)
    point_query = json.loads(point_query_str)

    # Should still produce valid query (might be treated as point by SAM3)
    assert box_query[0]["x1"] == 100.0
    assert box_query[0]["x2"] == 100.0  # x1 == x2
    assert point_query[0]["x"] == 100.0  # center at x

    # Zero height
    bbox = [[100, 200, 50, 0]]
    box_query_str, point_query_str = node.bbox_to_sam3_query(bbox)

    box_query = json.loads(box_query_str)
    point_query = json.loads(point_query_str)

    assert box_query[0]["y1"] == 200.0
    assert box_query[0]["y2"] == 200.0  # y1 == y2
    assert point_query[0]["y"] == 200.0  # center at y

    print("✓ test_zero_size_bbox passed")


def test_empty_bbox():
    """Test handling of empty bbox input"""
    node = BBoxToSAM3Query()

    # Empty list
    bbox = []
    box_query_str, point_query_str = node.bbox_to_sam3_query(bbox)

    assert box_query_str == "[]", "Should return empty array string"
    assert point_query_str == "[]", "Should return empty array string"

    print("✓ test_empty_bbox passed")


def test_invalid_bbox_length():
    """Test handling of bbox with wrong number of elements"""
    node = BBoxToSAM3Query()

    # Too few elements
    bbox = [[100, 200, 50]]  # Missing height
    box_query_str, point_query_str = node.bbox_to_sam3_query(bbox)

    assert box_query_str == "[]", "Should return empty array for invalid bbox"
    assert point_query_str == "[]", "Should return empty array for invalid bbox"

    # Too many elements
    bbox = [[100, 200, 50, 75, 999]]  # Extra element
    box_query_str, point_query_str = node.bbox_to_sam3_query(bbox)

    assert box_query_str == "[]", "Should return empty array for invalid bbox"
    assert point_query_str == "[]", "Should return empty array for invalid bbox"

    print("✓ test_invalid_bbox_length passed")


def test_negative_coordinates():
    """Test handling of negative coordinates (edge case)"""
    node = BBoxToSAM3Query()

    # Negative coordinates (might happen with cropped images)
    bbox = [[-10, -20, 50, 75]]

    box_query_str, point_query_str = node.bbox_to_sam3_query(bbox)

    box_query = json.loads(box_query_str)
    point_query = json.loads(point_query_str)

    # Should still convert (downstream node handles clipping)
    assert box_query[0]["x1"] == -10.0
    assert box_query[0]["y1"] == -20.0
    assert box_query[0]["x2"] == 40.0  # -10 + 50
    assert box_query[0]["y2"] == 55.0  # -20 + 75

    assert point_query[0]["x"] == 15.0  # -10 + 50/2
    assert point_query[0]["y"] == 17.5  # -20 + 75/2

    print("✓ test_negative_coordinates passed")


def test_json_format_validation():
    """Test that output is valid JSON with correct structure"""
    node = BBoxToSAM3Query()

    bbox = [[100, 200, 50, 75]]
    box_query_str, point_query_str = node.bbox_to_sam3_query(bbox)

    # Should parse without errors
    box_query = json.loads(box_query_str)
    point_query = json.loads(point_query_str)

    # Check structure
    assert isinstance(box_query, list), "Box query should be a list"
    assert isinstance(box_query[0], dict), "Box query item should be a dict"
    assert set(box_query[0].keys()) == {"x1", "y1", "x2", "y2"}, "Box query should have x1, y1, x2, y2 keys"

    assert isinstance(point_query, list), "Point query should be a list"
    assert isinstance(point_query[0], dict), "Point query item should be a dict"
    assert set(point_query[0].keys()) == {"x", "y"}, "Point query should have x, y keys"

    print("✓ test_json_format_validation passed")


def test_float_precision():
    """Test that float values maintain reasonable precision"""
    node = BBoxToSAM3Query()

    # Use values that would have decimal places
    bbox = [[100.5, 200.3, 50.7, 75.9]]

    box_query_str, point_query_str = node.bbox_to_sam3_query(bbox)

    box_query = json.loads(box_query_str)
    point_query = json.loads(point_query_str)

    # Check precision is preserved (use approximate equality for floats)
    assert box_query[0]["x1"] == 100.5
    assert box_query[0]["y1"] == 200.3
    assert abs(box_query[0]["x2"] - 151.2) < 0.0001  # 100.5 + 50.7
    assert abs(box_query[0]["y2"] - 276.2) < 0.0001  # 200.3 + 75.9

    assert abs(point_query[0]["x"] - 125.85) < 0.0001  # 100.5 + 50.7/2
    assert abs(point_query[0]["y"] - 238.25) < 0.0001  # 200.3 + 75.9/2

    print("✓ test_float_precision passed")


def test_large_coordinates():
    """Test handling of large coordinate values"""
    node = BBoxToSAM3Query()

    # Large image coordinates (e.g., 4K image)
    bbox = [[1920, 1080, 1024, 768]]

    box_query_str, point_query_str = node.bbox_to_sam3_query(bbox)

    box_query = json.loads(box_query_str)
    point_query = json.loads(point_query_str)

    # Check conversions work with large values
    assert box_query[0]["x1"] == 1920.0
    assert box_query[0]["y1"] == 1080.0
    assert box_query[0]["x2"] == 2944.0  # 1920 + 1024
    assert box_query[0]["y2"] == 1848.0  # 1080 + 768

    assert point_query[0]["x"] == 2432.0  # 1920 + 1024/2
    assert point_query[0]["y"] == 1464.0  # 1080 + 768/2

    print("✓ test_large_coordinates passed")


def test_input_types_signature():
    """Test that INPUT_TYPES matches function signature"""
    node = BBoxToSAM3Query()

    input_types = node.INPUT_TYPES()

    # Check required inputs
    assert "required" in input_types
    assert "bbox" in input_types["required"]
    assert input_types["required"]["bbox"][0] == "BBOX"

    # Check no optional inputs
    assert "optional" not in input_types or len(input_types["optional"]) == 0

    print("✓ test_input_types_signature passed")


def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*50)
    print("Testing BBox to SAM3 Query Node")
    print("="*50 + "\n")

    tests = [
        test_basic_conversion,
        test_unwrapped_bbox_format,
        test_integer_coordinates,
        test_zero_size_bbox,
        test_empty_bbox,
        test_invalid_bbox_length,
        test_negative_coordinates,
        test_json_format_validation,
        test_float_precision,
        test_large_coordinates,
        test_input_types_signature,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1

    print("\n" + "="*50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*50 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
