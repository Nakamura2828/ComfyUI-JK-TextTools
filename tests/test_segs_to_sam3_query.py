"""
Tests for SEGs to SAM3 Query Node

Test the conversion of SEGS format to SAM3 Selector query formats.
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from segs_to_sam3_query import SEGsToSAM3Query


class MockSEG:
    """Mock SEG object for testing"""
    def __init__(self, cropped_mask, crop_region, label="", confidence=1.0):
        self.cropped_mask = cropped_mask
        self.crop_region = crop_region
        self.label = label
        self.confidence = confidence


def create_mock_segs(height, width, seg_data):
    """
    Create mock SEGS tuple for testing.

    Args:
        height: Image height
        width: Image width
        seg_data: List of tuples (mask, crop_region, label, confidence)

    Returns:
        SEGS tuple ((height, width), [SEG(...), ...])
    """
    segs_list = []
    for mask, crop_region, label, confidence in seg_data:
        seg = MockSEG(mask, crop_region, label, confidence)
        segs_list.append(seg)

    return ((height, width), segs_list)


def test_basic_conversion():
    """Test basic SEGS to SAM3 query conversion with all four outputs"""
    node = SEGsToSAM3Query()

    # Create a simple rectangular mask
    mask = np.ones((50, 50), dtype=np.float32)
    seg_data = [
        (mask, [10, 20, 60, 70], "person", 0.95)
    ]

    segs = create_mock_segs(256, 256, seg_data)
    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.segs_to_sam3_query(segs)

    # === Check SAM3 format outputs (normalized coordinates) ===
    # Box prompt should have normalized XYWH
    assert "boxes" in box_sam3, "SAM3 box should have 'boxes' key"
    assert "labels" in box_sam3, "SAM3 box should have 'labels' key"
    assert len(box_sam3["boxes"]) == 1, "Should have one box"
    assert len(box_sam3["labels"]) == 1, "Should have one label"

    # Verify normalized coordinates (bbox is from 10,20 to 59,69)
    x_norm, y_norm, w_norm, h_norm = box_sam3["boxes"][0]
    assert abs(x_norm - 10.0/256) < 0.001, f"x should be normalized, got {x_norm}"
    assert abs(y_norm - 20.0/256) < 0.001, f"y should be normalized, got {y_norm}"
    assert abs(w_norm - 49.0/256) < 0.001, f"w should be normalized, got {w_norm}"  # 59-10
    assert abs(h_norm - 49.0/256) < 0.001, f"h should be normalized, got {h_norm}"  # 69-20

    # Verify positive label (default)
    assert box_sam3["labels"][0] == True, "Default should be positive (True)"

    # Point prompt should have normalized coordinates
    assert "points" in point_sam3, "SAM3 point should have 'points' key"
    assert "labels" in point_sam3, "SAM3 point should have 'labels' key"
    assert point_sam3["labels"][0] == 1, "Default should be positive (1)"

    # === Check TBG format outputs (absolute coordinates) ===
    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Check box query (should be the crop_region bounds)
    assert len(box_tbg) == 1, "Should have one box"
    assert box_tbg[0]["x1"] == 10.0, f"x1 should be 10, got {box_tbg[0]['x1']}"
    assert box_tbg[0]["y1"] == 20.0, f"y1 should be 20, got {box_tbg[0]['y1']}"
    assert box_tbg[0]["x2"] == 59.0, f"x2 should be 59 (max x), got {box_tbg[0]['x2']}"
    assert box_tbg[0]["y2"] == 69.0, f"y2 should be 69 (max y), got {box_tbg[0]['y2']}"

    # Check point query (should be centroid of the mask)
    assert len(point_tbg) == 1, "Should have one point"
    # Centroid should be near the center of the crop region
    expected_x = (10 + 59) / 2.0  # ~34.5
    expected_y = (20 + 69) / 2.0  # ~44.5
    assert abs(point_tbg[0]["x"] - expected_x) < 1.0, f"centroid_x should be ~{expected_x}, got {point_tbg[0]['x']}"
    assert abs(point_tbg[0]["y"] - expected_y) < 1.0, f"centroid_y should be ~{expected_y}, got {point_tbg[0]['y']}"

    print("✓ test_basic_conversion passed")


def test_multiple_segments_union():
    """Test union of multiple segments"""
    node = SEGsToSAM3Query()

    # Create two separate masks
    mask1 = np.ones((30, 30), dtype=np.float32)
    mask2 = np.ones((40, 40), dtype=np.float32)

    seg_data = [
        (mask1, [10, 10, 40, 40], "person_0", 0.95),
        (mask2, [100, 100, 140, 140], "person_1", 0.87)
    ]

    segs = create_mock_segs(256, 256, seg_data)
    _, _, box_tbg_str, point_tbg_str = node.segs_to_sam3_query(segs)

    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Box should encompass both segments
    assert box_tbg[0]["x1"] == 10.0, "x1 should be min of all segments"
    assert box_tbg[0]["y1"] == 10.0, "y1 should be min of all segments"
    assert box_tbg[0]["x2"] == 139.0, "x2 should be max of all segments"
    assert box_tbg[0]["y2"] == 139.0, "y2 should be max of all segments"

    # Centroid should be between the two masks
    # With equal area masks, centroid should be roughly between them
    assert 10 <= point_tbg[0]["x"] <= 140, "centroid_x should be within bbox"
    assert 10 <= point_tbg[0]["y"] <= 140, "centroid_y should be within bbox"

    print("✓ test_multiple_segments_union passed")


def test_centroid_calculation():
    """Test that centroid is calculated correctly for non-uniform masks"""
    node = SEGsToSAM3Query()

    # Create an L-shaped mask (non-convex)
    mask = np.zeros((50, 50), dtype=np.float32)
    mask[0:25, 0:10] = 1.0  # Vertical bar
    mask[0:10, 0:50] = 1.0  # Horizontal bar (L-shape)

    seg_data = [
        (mask, [100, 100, 150, 150], "object", 0.9)
    ]

    segs = create_mock_segs(256, 256, seg_data)
    _, _, box_tbg_str, point_tbg_str = node.segs_to_sam3_query(segs)

    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Box should be full crop region
    assert box_tbg[0]["x1"] == 100.0
    assert box_tbg[0]["x2"] == 149.0  # max coordinate (150 - 1)

    # Centroid should be different from bbox center (due to L-shape)
    bbox_center_x = (100 + 149) / 2.0
    bbox_center_y = (100 + 149) / 2.0

    # Centroid should be shifted towards the heavier part of the L
    # (top-left area has more pixels)
    centroid_x = point_tbg[0]["x"]
    centroid_y = point_tbg[0]["y"]

    # Centroid should be within the bbox
    assert 100 <= centroid_x <= 150, f"centroid_x {centroid_x} should be within bbox"
    assert 100 <= centroid_y <= 150, f"centroid_y {centroid_y} should be within bbox"

    # For L-shape with horizontal bar wider, centroid_y should be closer to top
    assert centroid_y < bbox_center_y, "centroid_y should be above center for top-heavy L-shape"

    print("✓ test_centroid_calculation passed")


def test_empty_segs():
    """Test handling of empty SEGS"""
    node = SEGsToSAM3Query()

    # Empty seg list
    segs = ((256, 256), [])
    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.segs_to_sam3_query(segs)

    # SAM3 outputs should be empty
    assert box_sam3["boxes"] == [], "SAM3 box should be empty"
    assert box_sam3["labels"] == [], "SAM3 labels should be empty"
    assert point_sam3["points"] == [], "SAM3 point should be empty"
    assert point_sam3["labels"] == [], "SAM3 labels should be empty"

    # TBG outputs should be empty
    assert box_tbg_str == "[]", "Should return empty array for empty segs"
    assert point_tbg_str == "[]", "Should return empty array for empty segs"

    print("✓ test_empty_segs passed")


def test_invalid_segs_format():
    """Test handling of invalid SEGS format"""
    node = SEGsToSAM3Query()

    # Not a tuple
    segs = "invalid"
    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.segs_to_sam3_query(segs)
    assert box_sam3["boxes"] == [], "SAM3 outputs should be empty for invalid format"
    assert box_tbg_str == "[]", "Should return empty array for invalid format"
    assert point_tbg_str == "[]", "Should return empty array for invalid format"

    # Wrong tuple length
    segs = ((256, 256),)
    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.segs_to_sam3_query(segs)
    assert box_sam3["boxes"] == [], "SAM3 outputs should be empty for wrong tuple length"
    assert box_tbg_str == "[]", "Should return empty array for wrong tuple length"

    # Invalid dimensions
    segs = ("invalid", [])
    box_sam3, point_sam3, box_tbg_str, point_tbg_str = node.segs_to_sam3_query(segs)
    assert box_sam3["boxes"] == [], "SAM3 outputs should be empty for invalid dimensions"
    assert box_tbg_str == "[]", "Should return empty array for invalid dimensions"

    print("✓ test_invalid_segs_format passed")


def test_none_cropped_mask():
    """Test handling of segments with None cropped_mask"""
    node = SEGsToSAM3Query()

    # One valid seg, one with None mask
    mask = np.ones((30, 30), dtype=np.float32)
    seg_data = [
        (None, [10, 10, 40, 40], "invalid", 0.9),
        (mask, [50, 50, 80, 80], "valid", 0.95)
    ]

    segs = create_mock_segs(256, 256, seg_data)
    _, _, box_tbg_str, _ = node.segs_to_sam3_query(segs)

    box_tbg = json.loads(box_tbg_str)

    # Should only include the valid segment
    assert box_tbg[0]["x1"] == 50.0, "Should ignore None mask segment"
    assert box_tbg[0]["y1"] == 50.0

    print("✓ test_none_cropped_mask passed")


def test_coordinate_clamping():
    """Test that coordinates are clamped to image bounds"""
    node = SEGsToSAM3Query()

    # Mask that extends beyond image bounds
    mask = np.ones((100, 100), dtype=np.float32)
    seg_data = [
        (mask, [200, 200, 300, 300], "object", 0.9)  # Extends beyond 256x256
    ]

    segs = create_mock_segs(256, 256, seg_data)
    _, _, box_tbg_str, point_tbg_str = node.segs_to_sam3_query(segs)

    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Coordinates should be clamped to 0-255
    assert box_tbg[0]["x2"] <= 255.0, "x2 should be clamped to image width"
    assert box_tbg[0]["y2"] <= 255.0, "y2 should be clamped to image height"
    assert point_tbg[0]["x"] <= 256.0, "centroid_x should be within bounds"
    assert point_tbg[0]["y"] <= 256.0, "centroid_y should be within bounds"

    print("✓ test_coordinate_clamping passed")


def test_tensor_mask_input():
    """Test handling of PyTorch tensor masks (not numpy)"""
    node = SEGsToSAM3Query()

    # Use torch tensor instead of numpy array
    mask = torch.ones((40, 40), dtype=torch.float32)
    seg_data = [
        (mask, [20, 20, 60, 60], "object", 0.9)
    ]

    segs = create_mock_segs(256, 256, seg_data)
    _, _, box_tbg_str, point_tbg_str = node.segs_to_sam3_query(segs)

    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Should work with torch tensors
    assert len(box_tbg) == 1, "Should handle torch tensor masks"
    assert len(point_tbg) == 1, "Should handle torch tensor masks"
    assert box_tbg[0]["x1"] == 20.0

    print("✓ test_tensor_mask_input passed")


def test_binary_vs_float_mask():
    """Test that both binary and float masks work"""
    node = SEGsToSAM3Query()

    # Float mask with values between 0 and 1
    mask = np.random.rand(40, 40).astype(np.float32)
    mask[mask < 0.5] = 0  # Some zeros, some floats > 0.5

    seg_data = [
        (mask, [30, 30, 70, 70], "object", 0.9)
    ]

    segs = create_mock_segs(256, 256, seg_data)
    _, _, box_tbg_str, point_tbg_str = node.segs_to_sam3_query(segs)

    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Should handle float masks (threshold at 0.5)
    assert len(box_tbg) == 1, "Should handle float masks"
    assert len(point_tbg) == 1, "Should handle float masks"

    print("✓ test_binary_vs_float_mask passed")


def test_json_format_validation():
    """Test that TBG output is valid JSON with correct structure"""
    node = SEGsToSAM3Query()

    mask = np.ones((40, 40), dtype=np.float32)
    seg_data = [
        (mask, [50, 60, 90, 100], "object", 0.9)
    ]

    segs = create_mock_segs(256, 256, seg_data)
    _, _, box_tbg_str, point_tbg_str = node.segs_to_sam3_query(segs)

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

    # All values should be floats
    assert all(isinstance(v, float) for v in box_tbg[0].values()), "Box query values should be floats"
    assert all(isinstance(v, float) for v in point_tbg[0].values()), "Point query values should be floats"

    print("✓ test_json_format_validation passed")


def test_large_image():
    """Test handling of large image dimensions"""
    node = SEGsToSAM3Query()

    # 4K resolution
    mask = np.ones((100, 100), dtype=np.float32)
    seg_data = [
        (mask, [1920, 1080, 2020, 1180], "object", 0.9)
    ]

    segs = create_mock_segs(2160, 3840, seg_data)
    _, _, box_tbg_str, point_tbg_str = node.segs_to_sam3_query(segs)

    box_tbg = json.loads(box_tbg_str)
    point_tbg = json.loads(point_tbg_str)

    # Should handle large coordinates
    assert box_tbg[0]["x1"] >= 1900, "Should handle large image coordinates"
    assert box_tbg[0]["y1"] >= 1000
    assert point_tbg[0]["x"] >= 1900

    print("✓ test_large_image passed")


def test_positive_prompt_type():
    """Test positive prompt type labeling"""
    node = SEGsToSAM3Query()

    mask = np.ones((40, 40), dtype=np.float32)
    seg_data = [(mask, [50, 60, 90, 100], "object", 0.9)]
    segs = create_mock_segs(256, 256, seg_data)

    box_sam3, point_sam3, _, _ = node.segs_to_sam3_query(segs, prompt_type="positive")

    # Check positive labels
    assert box_sam3["labels"][0] == True, "Positive box label should be True"
    assert point_sam3["labels"][0] == 1, "Positive point label should be 1"

    print("✓ test_positive_prompt_type passed")


def test_negative_prompt_type():
    """Test negative prompt type labeling"""
    node = SEGsToSAM3Query()

    mask = np.ones((40, 40), dtype=np.float32)
    seg_data = [(mask, [50, 60, 90, 100], "object", 0.9)]
    segs = create_mock_segs(256, 256, seg_data)

    box_sam3, point_sam3, _, _ = node.segs_to_sam3_query(segs, prompt_type="negative")

    # Check negative labels
    assert box_sam3["labels"][0] == False, "Negative box label should be False"
    assert point_sam3["labels"][0] == 0, "Negative point label should be 0"

    print("✓ test_negative_prompt_type passed")


def test_sam3_format_structure():
    """Test SAM3 format matches expected structure"""
    node = SEGsToSAM3Query()

    mask = np.ones((40, 40), dtype=np.float32)
    seg_data = [(mask, [50, 60, 90, 100], "object", 0.9)]
    segs = create_mock_segs(256, 256, seg_data)

    box_sam3, point_sam3, _, _ = node.segs_to_sam3_query(segs)

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

    # Verify normalized coordinates are in 0-1 range
    assert all(0 <= v <= 1 for v in box_sam3["boxes"][0]), "SAM3 coordinates should be normalized 0-1"
    assert all(0 <= v <= 1 for v in point_sam3["points"][0]), "SAM3 coordinates should be normalized 0-1"

    print("✓ test_sam3_format_structure passed")


def test_input_types_signature():
    """Test that INPUT_TYPES matches function signature"""
    node = SEGsToSAM3Query()

    input_types = node.INPUT_TYPES()

    # Check required inputs
    assert "required" in input_types
    assert "segs" in input_types["required"]
    assert input_types["required"]["segs"][0] == "SEGS"

    # Check optional inputs
    assert "optional" in input_types
    assert "prompt_type" in input_types["optional"]

    # Check return types
    assert node.RETURN_TYPES == ("SAM3_BOXES_PROMPT", "SAM3_POINTS_PROMPT", "STRING", "STRING")
    assert node.RETURN_NAMES == ("box_sam3", "point_sam3", "box_tbg_sam3", "point_tbg_sam3")

    print("✓ test_input_types_signature passed")


def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*50)
    print("Testing SEGs to SAM3 Query Node")
    print("="*50 + "\n")

    tests = [
        test_basic_conversion,
        test_multiple_segments_union,
        test_centroid_calculation,
        test_empty_segs,
        test_invalid_segs_format,
        test_none_cropped_mask,
        test_coordinate_clamping,
        test_tensor_mask_input,
        test_binary_vs_float_mask,
        test_json_format_validation,
        test_large_image,
        test_positive_prompt_type,
        test_negative_prompt_type,
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
