"""
Tests for Mask to BBox Node

Test the conversion of masks to bounding boxes in XYWH format.
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mask_to_bbox import MaskToBBox


def test_basic_rectangular_mask():
    """Test conversion of simple rectangular mask"""
    node = MaskToBBox()

    # Create a 256x256 mask with a rectangle from (50,60) to (100,120)
    mask = torch.zeros((256, 256), dtype=torch.float32)
    mask[60:121, 50:101] = 1.0

    bbox, x, y, w, h = node.mask_to_bbox(mask)

    # Check bbox format
    assert isinstance(bbox, list), "BBox should be a list"
    assert len(bbox) == 1, "BBox should have one entry"
    assert len(bbox[0]) == 4, "BBox entry should have 4 values"

    # Check coordinates
    assert x == 50, f"x should be 50, got {x}"
    assert y == 60, f"y should be 60, got {y}"
    assert w == 51, f"w should be 51 (inclusive), got {w}"
    assert h == 61, f"h should be 61 (inclusive), got {h}"

    # Check bbox matches individual coordinates
    assert bbox[0] == [x, y, w, h], "BBox should match individual coordinates"

    # Check all are integers
    assert isinstance(x, int), "x should be int"
    assert isinstance(y, int), "y should be int"
    assert isinstance(w, int), "w should be int"
    assert isinstance(h, int), "h should be int"

    print("✓ test_basic_rectangular_mask passed")


def test_single_pixel():
    """Test mask with single pixel"""
    node = MaskToBBox()

    mask = torch.zeros((100, 100), dtype=torch.float32)
    mask[50, 40] = 1.0

    bbox, x, y, w, h = node.mask_to_bbox(mask)

    assert x == 40, "Single pixel x"
    assert y == 50, "Single pixel y"
    assert w == 1, "Single pixel width should be 1"
    assert h == 1, "Single pixel height should be 1"

    print("✓ test_single_pixel passed")


def test_irregular_shape():
    """Test mask with irregular shape (L-shape)"""
    node = MaskToBBox()

    mask = torch.zeros((100, 100), dtype=torch.float32)
    # L-shape: vertical bar + horizontal bar
    mask[10:50, 20:25] = 1.0  # Vertical bar
    mask[45:50, 20:60] = 1.0  # Horizontal bar

    bbox, x, y, w, h = node.mask_to_bbox(mask)

    # BBox should encompass entire L-shape
    assert x == 20, f"x should be 20 (leftmost), got {x}"
    assert y == 10, f"y should be 10 (topmost), got {y}"
    assert w == 40, f"w should be 40 (20 to 59), got {w}"
    assert h == 40, f"h should be 40 (10 to 49), got {h}"

    print("✓ test_irregular_shape passed")


def test_float_mask_threshold():
    """Test that float mask values are thresholded at 0.5"""
    node = MaskToBBox()

    mask = torch.zeros((100, 100), dtype=torch.float32)
    # Values below 0.5 should be ignored
    mask[10:20, 10:20] = 0.3
    # Values above 0.5 should be included
    mask[30:40, 30:40] = 0.7

    bbox, x, y, w, h = node.mask_to_bbox(mask)

    # Should only include the 0.7 region
    assert x == 30, "Should threshold at 0.5"
    assert y == 30
    assert w == 10
    assert h == 10

    print("✓ test_float_mask_threshold passed")


def test_empty_mask():
    """Test handling of empty mask"""
    node = MaskToBBox()

    mask = torch.zeros((100, 100), dtype=torch.float32)

    bbox, x, y, w, h = node.mask_to_bbox(mask)

    # Should return zero bbox
    assert x == 0, "Empty mask should return zero bbox"
    assert y == 0
    assert w == 0
    assert h == 0
    assert bbox == [[0, 0, 0, 0]]

    print("✓ test_empty_mask passed")


def test_full_mask():
    """Test mask covering entire image"""
    node = MaskToBBox()

    mask = torch.ones((128, 256), dtype=torch.float32)

    bbox, x, y, w, h = node.mask_to_bbox(mask)

    assert x == 0, "Full mask x should start at 0"
    assert y == 0, "Full mask y should start at 0"
    assert w == 256, "Full mask width should be image width"
    assert h == 128, "Full mask height should be image height"

    print("✓ test_full_mask passed")


def test_batched_mask():
    """Test handling of batched mask (3D tensor)"""
    node = MaskToBBox()

    # Create batched mask (batch_size=3, height=100, width=100)
    mask = torch.zeros((3, 100, 100), dtype=torch.float32)
    # Only first mask has content
    mask[0, 20:40, 30:50] = 1.0

    bbox, x, y, w, h = node.mask_to_bbox(mask)

    # Should use first mask in batch
    assert x == 30, "Should use first mask in batch"
    assert y == 20
    assert w == 20
    assert h == 20

    print("✓ test_batched_mask passed")


def test_edge_cases_coordinates():
    """Test edge cases with coordinates at image boundaries"""
    node = MaskToBBox()

    # Mask at top-left corner
    mask = torch.zeros((100, 100), dtype=torch.float32)
    mask[0:10, 0:10] = 1.0

    bbox, x, y, w, h = node.mask_to_bbox(mask)

    assert x == 0, "Corner mask x"
    assert y == 0, "Corner mask y"
    assert w == 10, "Corner mask width"
    assert h == 10, "Corner mask height"

    # Mask at bottom-right corner
    mask = torch.zeros((100, 100), dtype=torch.float32)
    mask[90:100, 90:100] = 1.0

    bbox, x, y, w, h = node.mask_to_bbox(mask)

    assert x == 90, "Bottom-right x"
    assert y == 90, "Bottom-right y"
    assert w == 10, "Bottom-right width"
    assert h == 10, "Bottom-right height"

    print("✓ test_edge_cases_coordinates passed")


def test_invalid_mask_types():
    """Test handling of invalid mask inputs"""
    node = MaskToBBox()

    # Not a tensor
    bbox, x, y, w, h = node.mask_to_bbox("invalid")
    assert bbox == [[0, 0, 0, 0]], "Should handle non-tensor input"

    # Wrong dimensions (4D)
    mask = torch.zeros((1, 1, 100, 100), dtype=torch.float32)
    bbox, x, y, w, h = node.mask_to_bbox(mask)
    assert bbox == [[0, 0, 0, 0]], "Should handle wrong dimensions"

    # 1D tensor
    mask = torch.zeros(100, dtype=torch.float32)
    bbox, x, y, w, h = node.mask_to_bbox(mask)
    assert bbox == [[0, 0, 0, 0]], "Should handle 1D tensor"

    print("✓ test_invalid_mask_types passed")


def test_chain_to_bbox_to_sam3():
    """Test that output can chain to BBox to SAM3 Query"""
    node = MaskToBBox()

    mask = torch.zeros((256, 256), dtype=torch.float32)
    mask[50:100, 60:110] = 1.0

    bbox, x, y, w, h = node.mask_to_bbox(mask)

    # BBox format should be compatible with BBox to SAM3 Query
    assert isinstance(bbox, list), "Should be list"
    assert isinstance(bbox[0], list), "Should be nested list"
    assert len(bbox[0]) == 4, "Should have 4 coordinates"
    assert all(isinstance(coord, int) for coord in bbox[0]), "All coords should be ints"

    print("✓ test_chain_to_bbox_to_sam3 passed")


def test_return_types():
    """Test that return types match INPUT_TYPES specification"""
    node = MaskToBBox()

    mask = torch.ones((50, 50), dtype=torch.float32)
    bbox, x, y, w, h = node.mask_to_bbox(mask)

    # Check types match RETURN_TYPES specification
    assert isinstance(bbox, list), "BBox should be list (BBOX type)"
    assert isinstance(x, int), "x should be INT"
    assert isinstance(y, int), "y should be INT"
    assert isinstance(w, int), "w should be INT"
    assert isinstance(h, int), "h should be INT"

    print("✓ test_return_types passed")


def test_input_types_signature():
    """Test that INPUT_TYPES matches function signature"""
    node = MaskToBBox()

    input_types = node.INPUT_TYPES()

    # Check required inputs
    assert "required" in input_types
    assert "mask" in input_types["required"]
    assert input_types["required"]["mask"][0] == "MASK"

    # Check no optional inputs
    assert "optional" not in input_types or len(input_types["optional"]) == 0

    print("✓ test_input_types_signature passed")


def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*50)
    print("Testing Mask to BBox Node")
    print("="*50 + "\n")

    tests = [
        test_basic_rectangular_mask,
        test_single_pixel,
        test_irregular_shape,
        test_float_mask_threshold,
        test_empty_mask,
        test_full_mask,
        test_batched_mask,
        test_edge_cases_coordinates,
        test_invalid_mask_types,
        test_chain_to_bbox_to_sam3,
        test_return_types,
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
