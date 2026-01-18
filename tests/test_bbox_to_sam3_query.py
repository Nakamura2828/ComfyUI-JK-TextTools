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
    """Test basic conversion with all four outputs (SAM3 + TBG)"""
    node = BBoxToSAM3Query()

    # BBOX: x=100, y=200, w=50, h=75
    # Image: 512x512
    bbox = [[100, 200, 50, 75]]
    width = 512
    height = 512

    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox, width, height)

    # === Check SAM3 format outputs (normalized coordinates) ===
    # Box prompt should have normalized XYWH
    assert "boxes" in box_sam3, "SAM3 box should have 'boxes' key"
    assert "labels" in box_sam3, "SAM3 box should have 'labels' key"
    assert len(box_sam3["boxes"]) == 1, "Should have one box"
    assert len(box_sam3["labels"]) == 1, "Should have one label"

    # Verify normalized coordinates
    x_norm, y_norm, w_norm, h_norm = box_sam3["boxes"][0]
    assert abs(x_norm - 100.0/512) < 0.0001, "x should be normalized"
    assert abs(y_norm - 200.0/512) < 0.0001, "y should be normalized"
    assert abs(w_norm - 50.0/512) < 0.0001, "w should be normalized"
    assert abs(h_norm - 75.0/512) < 0.0001, "h should be normalized"

    # Verify positive label (default)
    assert box_sam3["labels"][0] == True, "Default should be positive (True)"

    # Point prompt should have normalized coordinates
    assert "points" in point_sam3, "SAM3 point should have 'points' key"
    assert "labels" in point_sam3, "SAM3 point should have 'labels' key"
    assert len(point_sam3["points"]) == 1, "Should have one point"
    assert len(point_sam3["labels"]) == 1, "Should have one label"

    # Verify normalized center point
    center_x_norm, center_y_norm = point_sam3["points"][0]
    expected_x = (100 + 50/2) / 512  # 125/512
    expected_y = (200 + 75/2) / 512  # 237.5/512
    assert abs(center_x_norm - expected_x) < 0.0001, "center_x should be normalized"
    assert abs(center_y_norm - expected_y) < 0.0001, "center_y should be normalized"

    # Verify positive label (default)
    assert point_sam3["labels"][0] == 1, "Default should be positive (1)"

    # === Check TBG format outputs (absolute coordinates, JSON strings) ===
    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Check box query (XYXY format)
    assert len(box_tbg) == 1, "Should have one box"
    assert box_tbg[0]["x1"] == 100.0, "x1 should be x"
    assert box_tbg[0]["y1"] == 200.0, "y1 should be y"
    assert box_tbg[0]["x2"] == 150.0, "x2 should be x + w"
    assert box_tbg[0]["y2"] == 275.0, "y2 should be y + h"

    # Check point query (center point)
    assert len(point_tbg) == 1, "Should have one point"
    assert point_tbg[0]["x"] == 125.0, "center_x should be x + w/2"
    assert point_tbg[0]["y"] == 237.5, "center_y should be y + h/2"

    print("✓ test_basic_conversion passed")


def test_unwrapped_bbox_format():
    """Test handling of unwrapped bbox format [x,y,w,h]"""
    node = BBoxToSAM3Query()

    # Unwrapped format (not nested)
    bbox = [50, 60, 100, 120]

    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox, 640, 480)

    # Check SAM3 normalized outputs
    assert len(box_sam3["boxes"]) == 1, "Should handle unwrapped format"
    assert abs(box_sam3["boxes"][0][0] - 50.0/640) < 0.0001

    # Check TBG outputs
    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    assert box_tbg[0]["x1"] == 50.0
    assert box_tbg[0]["y1"] == 60.0
    assert box_tbg[0]["x2"] == 150.0  # 50 + 100
    assert box_tbg[0]["y2"] == 180.0  # 60 + 120

    assert point_tbg[0]["x"] == 100.0  # 50 + 100/2
    assert point_tbg[0]["y"] == 120.0  # 60 + 120/2

    print("✓ test_unwrapped_bbox_format passed")


def test_integer_coordinates():
    """Test that integer coordinates are converted to floats"""
    node = BBoxToSAM3Query()

    # Pure integer coordinates
    bbox = [[10, 20, 30, 40]]

    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox, 512, 512)

    # Check SAM3 outputs have float values
    assert all(isinstance(v, float) for v in box_sam3["boxes"][0]), "SAM3 box coordinates should be floats"
    assert all(isinstance(v, float) for v in point_sam3["points"][0]), "SAM3 point coordinates should be floats"

    # Check TBG outputs
    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Verify all values are floats in the output
    assert isinstance(box_tbg[0]["x1"], float)
    assert isinstance(box_tbg[0]["y1"], float)
    assert isinstance(box_tbg[0]["x2"], float)
    assert isinstance(box_tbg[0]["y2"], float)

    assert isinstance(point_tbg[0]["x"], float)
    assert isinstance(point_tbg[0]["y"], float)

    print("✓ test_integer_coordinates passed")


def test_zero_size_bbox():
    """Test handling of zero-width or zero-height bbox"""
    node = BBoxToSAM3Query()

    # Zero width
    bbox = [[100, 200, 0, 50]]
    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox, 512, 512)

    # Check SAM3 outputs
    assert box_sam3["boxes"][0][2] == 0.0, "SAM3 width should be 0"
    assert point_sam3["points"][0][0] == 100.0/512, "SAM3 center_x normalized"

    # Check TBG outputs
    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Should still produce valid query (might be treated as point by SAM3)
    assert box_tbg[0]["x1"] == 100.0
    assert box_tbg[0]["x2"] == 100.0  # x1 == x2
    assert point_tbg[0]["x"] == 100.0  # center at x

    # Zero height
    bbox = [[100, 200, 50, 0]]
    _, _, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox, 512, 512)

    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    assert box_tbg[0]["y1"] == 200.0
    assert box_tbg[0]["y2"] == 200.0  # y1 == y2
    assert point_tbg[0]["y"] == 200.0  # center at y

    print("✓ test_zero_size_bbox passed")


def test_empty_bbox():
    """Test handling of empty bbox input"""
    node = BBoxToSAM3Query()

    # Empty list
    bbox = []
    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox)

    # SAM3 outputs should be empty
    assert box_sam3["boxes"] == [], "SAM3 box should be empty"
    assert box_sam3["labels"] == [], "SAM3 box labels should be empty"
    assert point_sam3["points"] == [], "SAM3 point should be empty"
    assert point_sam3["labels"] == [], "SAM3 point labels should be empty"

    # TBG outputs should be empty
    assert box_tbg_str == "[]", "Should return empty array string"
    assert point_tbg_str == "[]", "Should return empty array string"

    print("✓ test_empty_bbox passed")


def test_invalid_bbox_length():
    """Test handling of bbox with wrong number of elements"""
    node = BBoxToSAM3Query()

    # Too few elements
    bbox = [[100, 200, 50]]  # Missing height
    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox)

    # SAM3 outputs should be empty
    assert box_sam3["boxes"] == [], "SAM3 outputs should be empty for invalid bbox"
    assert point_sam3["points"] == [], "SAM3 outputs should be empty for invalid bbox"

    # TBG outputs should be empty
    assert box_tbg_str == "[]", "Should return empty array for invalid bbox"
    assert point_tbg_str == "[]", "Should return empty array for invalid bbox"

    # Too many elements
    bbox = [[100, 200, 50, 75, 999]]  # Extra element
    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox)

    assert box_sam3["boxes"] == [], "SAM3 outputs should be empty for invalid bbox"
    assert box_tbg_str == "[]", "Should return empty array for invalid bbox"
    assert point_tbg_str == "[]", "Should return empty array for invalid bbox"

    print("✓ test_invalid_bbox_length passed")


def test_negative_coordinates():
    """Test handling of negative coordinates (edge case)"""
    node = BBoxToSAM3Query()

    # Negative coordinates (might happen with cropped images)
    bbox = [[-10, -20, 50, 75]]

    _, _, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox, 512, 512)

    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Should still convert (downstream node handles clipping)
    assert box_tbg[0]["x1"] == -10.0
    assert box_tbg[0]["y1"] == -20.0
    assert box_tbg[0]["x2"] == 40.0  # -10 + 50
    assert box_tbg[0]["y2"] == 55.0  # -20 + 75

    assert point_tbg[0]["x"] == 15.0  # -10 + 50/2
    assert point_tbg[0]["y"] == 17.5  # -20 + 75/2

    print("✓ test_negative_coordinates passed")


def test_json_format_validation():
    """Test that TBG output is valid JSON with correct structure"""
    node = BBoxToSAM3Query()

    bbox = [[100, 200, 50, 75]]
    _, _, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox, 512, 512)

    # Should parse without errors
    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Check structure
    assert isinstance(box_tbg, list), "Box query should be a list"
    assert isinstance(box_tbg[0], dict), "Box query item should be a dict"
    assert set(box_tbg[0].keys()) == {"x1", "y1", "x2", "y2"}, "Box query should have x1, y1, x2, y2 keys"

    assert isinstance(point_tbg, list), "Point query should be a list"
    assert isinstance(point_tbg[0], dict), "Point query item should be a dict"
    assert set(point_tbg[0].keys()) == {"x", "y"}, "Point query should have x, y keys"

    print("✓ test_json_format_validation passed")


def test_float_precision():
    """Test that float values maintain reasonable precision"""
    node = BBoxToSAM3Query()

    # Use values that would have decimal places
    bbox = [[100.5, 200.3, 50.7, 75.9]]

    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox, 512, 512)

    # Check SAM3 normalized precision
    assert all(isinstance(v, float) for v in box_sam3["boxes"][0]), "SAM3 coordinates should be floats"

    # Check TBG precision
    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Check precision is preserved (use approximate equality for floats)
    assert box_tbg[0]["x1"] == 100.5
    assert box_tbg[0]["y1"] == 200.3
    assert abs(box_tbg[0]["x2"] - 151.2) < 0.0001  # 100.5 + 50.7
    assert abs(box_tbg[0]["y2"] - 276.2) < 0.0001  # 200.3 + 75.9

    assert abs(point_tbg[0]["x"] - 125.85) < 0.0001  # 100.5 + 50.7/2
    assert abs(point_tbg[0]["y"] - 238.25) < 0.0001  # 200.3 + 75.9/2

    print("✓ test_float_precision passed")


def test_large_coordinates():
    """Test handling of large coordinate values"""
    node = BBoxToSAM3Query()

    # Large image coordinates (e.g., 4K image)
    bbox = [[1920, 1080, 1024, 768]]

    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox, 3840, 2160)

    # Check SAM3 normalization for large dimensions
    assert 0 <= box_sam3["boxes"][0][0] <= 1, "SAM3 normalized x should be in 0-1 range"
    assert 0 <= box_sam3["boxes"][0][1] <= 1, "SAM3 normalized y should be in 0-1 range"

    # Check TBG absolute coordinates
    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Check conversions work with large values
    assert box_tbg[0]["x1"] == 1920.0
    assert box_tbg[0]["y1"] == 1080.0
    assert box_tbg[0]["x2"] == 2944.0  # 1920 + 1024
    assert box_tbg[0]["y2"] == 1848.0  # 1080 + 768

    assert point_tbg[0]["x"] == 2432.0  # 1920 + 1024/2
    assert point_tbg[0]["y"] == 1464.0  # 1080 + 768/2

    print("✓ test_large_coordinates passed")


def test_positive_prompt_type():
    """Test positive prompt type labeling"""
    node = BBoxToSAM3Query()

    bbox = [[100, 100, 50, 50]]
    box_sam3, point_sam3, _, _ = node.bbox_to_sam3_query(bbox, 512, 512, prompt_type="positive")

    # Check positive labels
    assert box_sam3["labels"][0] == True, "Positive box label should be True"
    assert point_sam3["labels"][0] == 1, "Positive point label should be 1"

    print("✓ test_positive_prompt_type passed")


def test_negative_prompt_type():
    """Test negative prompt type labeling"""
    node = BBoxToSAM3Query()

    bbox = [[100, 100, 50, 50]]
    box_sam3, point_sam3, _, _ = node.bbox_to_sam3_query(bbox, 512, 512, prompt_type="negative")

    # Check negative labels
    assert box_sam3["labels"][0] == False, "Negative box label should be False"
    assert point_sam3["labels"][0] == 0, "Negative point label should be 0"

    print("✓ test_negative_prompt_type passed")


def test_optional_dimensions_missing():
    """Test behavior when width/height not provided"""
    node = BBoxToSAM3Query()

    bbox = [[100, 100, 50, 50]]

    # No width/height provided
    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox)

    # SAM3 outputs should be empty
    assert box_sam3["boxes"] == [], "SAM3 box should be empty without dimensions"
    assert box_sam3["labels"] == [], "SAM3 box labels should be empty"
    assert point_sam3["points"] == [], "SAM3 point should be empty without dimensions"
    assert point_sam3["labels"] == [], "SAM3 point labels should be empty"

    # TBG outputs should still work
    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)
    assert len(box_tbg) == 1, "TBG box should still work"
    assert len(point_tbg) == 1, "TBG point should still work"
    assert box_tbg[0]["x1"] == 100.0

    print("✓ test_optional_dimensions_missing passed")


def test_optional_dimensions_zero():
    """Test behavior when width/height are zero"""
    node = BBoxToSAM3Query()

    bbox = [[100, 100, 50, 50]]

    # Explicit zero dimensions
    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.bbox_to_sam3_query(bbox, 0, 0)

    # SAM3 outputs should be empty
    assert box_sam3["boxes"] == [], "SAM3 outputs should be empty with zero dimensions"
    assert point_sam3["points"] == [], "SAM3 outputs should be empty with zero dimensions"

    # TBG outputs should still work
    box_tbg = json.loads(box_tbg_str)
    assert len(box_tbg) == 1, "TBG outputs should work regardless of dimensions"

    print("✓ test_optional_dimensions_zero passed")


def test_sam3_format_structure():
    """Test SAM3 format matches expected structure"""
    node = BBoxToSAM3Query()

    bbox = [[100, 200, 50, 75]]
    box_sam3, point_sam3, _, _ = node.bbox_to_sam3_query(bbox, 512, 512)

    # Box prompt structure
    assert isinstance(box_sam3, dict), "Box prompt should be a dict"
    assert set(box_sam3.keys()) == {"boxes", "labels"}, "Box prompt should have 'boxes' and 'labels' keys"
    assert isinstance(box_sam3["boxes"], list), "boxes should be a list"
    assert isinstance(box_sam3["labels"], list), "labels should be a list"
    assert len(box_sam3["boxes"][0]) == 4, "Box should have 4 coordinates (XYWH)"

    # Point prompt structure
    assert isinstance(point_sam3, dict), "Point prompt should be a dict"
    assert set(point_sam3.keys()) == {"points", "labels"}, "Point prompt should have 'points' and 'labels' keys"
    assert isinstance(point_sam3["points"], list), "points should be a list"
    assert isinstance(point_sam3["labels"], list), "labels should be a list"
    assert len(point_sam3["points"][0]) == 2, "Point should have 2 coordinates (XY)"

    print("✓ test_sam3_format_structure passed")


def test_input_types_signature():
    """Test that INPUT_TYPES matches function signature"""
    node = BBoxToSAM3Query()

    input_types = node.INPUT_TYPES()

    # Check required inputs
    assert "required" in input_types
    assert "bbox" in input_types["required"]
    assert input_types["required"]["bbox"][0] == "BBOX"

    # Check optional inputs
    assert "optional" in input_types
    assert "width" in input_types["optional"]
    assert "height" in input_types["optional"]
    assert "prompt_type" in input_types["optional"]

    # Check return types
    assert node.RETURN_TYPES == ("SAM3_BOXES_PROMPT", "SAM3_POINTS_PROMPT", "STRING", "STRING")
    assert node.RETURN_NAMES == ("box_sam3", "point_sam3", "box_tbg_sam3", "point_tbg_sam3")

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
        test_positive_prompt_type,
        test_negative_prompt_type,
        test_optional_dimensions_missing,
        test_optional_dimensions_zero,
        test_sam3_format_structure,
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
