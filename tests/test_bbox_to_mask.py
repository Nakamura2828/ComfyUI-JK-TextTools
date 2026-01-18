"""
Tests for BBox to Mask Node

Tests the simplified single bbox to mask converter.
For testing combined/union masks, see test_bboxes_to_mask.py

Run from project root:
    python tests/test_bbox_to_mask.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import node modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import inspect
from bbox_to_mask import BBoxToMask


def test_single_bbox():
    """Test converting single bbox to mask"""
    node = BBoxToMask()

    # Single bbox: top-left 100x100 square
    bbox = [[10, 20, 100, 100]]

    result = node.bbox_to_mask(bbox, width=512, height=512)

    # Check return structure
    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 1, "Should return 1 item (mask)"

    mask = result[0]
    assert mask.shape == (512, 512), f"Mask should be 512x512, got {mask.shape}"
    assert isinstance(mask, torch.Tensor), "Mask should be tensor"

    # Check bbox area is 1, rest is 0
    assert mask[20:120, 10:110].sum() == 100 * 100, "Bbox area should be all 1s"
    assert mask[0:20, :].sum() == 0, "Area above bbox should be all 0s"
    assert mask[:, 0:10].sum() == 0, "Area left of bbox should be all 0s"

    print("✓ test_single_bbox passed")


def test_wrapped_bbox_format():
    """Test that node handles [[x,y,w,h]] format correctly"""
    node = BBoxToMask()

    # Standard wrapped format
    bbox = [[10, 20, 50, 50]]

    result = node.bbox_to_mask(bbox, width=256, height=256)
    mask = result[0]

    # Should fill the bbox area
    assert mask[20:70, 10:60].sum() == 50 * 50, "Should fill bbox area"

    print("✓ test_wrapped_bbox_format passed")


def test_unwrapped_bbox_format():
    """Test that node handles [x,y,w,h] format correctly"""
    node = BBoxToMask()

    # Unwrapped format (might come from some sources)
    bbox = [10, 20, 50, 50]

    result = node.bbox_to_mask(bbox, width=256, height=256)
    mask = result[0]

    # Should fill the bbox area
    assert mask[20:70, 10:60].sum() == 50 * 50, "Should handle unwrapped format"

    print("✓ test_unwrapped_bbox_format passed")


def test_bbox_clamping():
    """Test that bboxes outside image bounds are clamped"""
    node = BBoxToMask()

    # Bbox that extends beyond image
    bbox = [[-10, -10, 100, 100]]  # Partially outside

    result = node.bbox_to_mask(bbox, width=256, height=256)
    mask = result[0]

    # Should only fill the part that's inside
    # Should fill from (0,0) to (90,90) - the part that's inside
    assert mask[0:90, 0:90].sum() > 0, "Should fill inside portion"
    assert mask[0, 0] == 1.0, "Top-left should be filled"

    print("✓ test_bbox_clamping passed")


def test_bbox_completely_outside():
    """Test bbox completely outside image bounds"""
    node = BBoxToMask()

    # Bbox completely outside image
    bbox = [[300, 300, 50, 50]]  # Outside 256x256 image

    result = node.bbox_to_mask(bbox, width=256, height=256)
    mask = result[0]

    # Should produce empty mask
    assert mask.sum() == 0, "Bbox outside image should produce empty mask"

    print("✓ test_bbox_completely_outside passed")


def test_invert_mode():
    """Test inverted mask (bbox is black, rest is white)"""
    node = BBoxToMask()

    bbox = [[10, 10, 50, 50]]

    result = node.bbox_to_mask(bbox, width=256, height=256, invert=True)
    mask = result[0]

    # Bbox area should be 0 (black)
    assert mask[10:60, 10:60].sum() == 0, "Bbox area should be 0 when inverted"

    # Rest should be 1 (white)
    assert mask[0:10, 0:10].sum() > 0, "Area outside bbox should be 1"
    assert mask[0, 0] == 1.0, "Outside area should be 1"

    # Total white area should be image size minus bbox size
    expected_white = (256 * 256) - (50 * 50)
    assert mask.sum() == expected_white, "Inverted mask should have correct white area"

    print("✓ test_invert_mode passed")


def test_zero_size_bbox():
    """Test handling of zero or negative size bbox"""
    node = BBoxToMask()

    # Zero width bbox
    bbox = [[10, 10, 0, 50]]

    result = node.bbox_to_mask(bbox, width=256, height=256)
    mask = result[0]

    # Should produce empty mask
    assert mask.sum() == 0, "Zero-size bbox should produce empty mask"

    print("✓ test_zero_size_bbox passed")


def test_empty_bbox():
    """Test handling of empty bbox list"""
    node = BBoxToMask()

    bbox = []

    result = node.bbox_to_mask(bbox, width=256, height=256)
    mask = result[0]

    # Should return empty mask
    assert isinstance(mask, torch.Tensor), "Should return tensor"
    assert mask.sum() == 0, "Empty bbox should produce empty mask"

    print("✓ test_empty_bbox passed")


def test_invalid_bbox():
    """Test handling of invalid bbox (wrong length)"""
    node = BBoxToMask()

    # Bbox with wrong number of coordinates
    bbox = [[10, 20, 30]]  # Only 3 values instead of 4

    result = node.bbox_to_mask(bbox, width=256, height=256)
    mask = result[0]

    # Should return empty mask
    assert mask.sum() == 0, "Invalid bbox should produce empty mask"

    print("✓ test_invalid_bbox passed")


def test_different_image_sizes():
    """Test with various image dimensions"""
    node = BBoxToMask()

    # Small image
    bbox = [[5, 5, 10, 10]]
    result = node.bbox_to_mask(bbox, width=64, height=64)
    assert result[0].shape == (64, 64), "Should match small image size"

    # Large image
    result = node.bbox_to_mask(bbox, width=2048, height=2048)
    assert result[0].shape == (2048, 2048), "Should match large image size"

    # Non-square
    result = node.bbox_to_mask(bbox, width=1920, height=1080)
    assert result[0].shape == (1080, 1920), "Should match non-square dimensions"

    print("✓ test_different_image_sizes passed")


def test_float_coordinates():
    """Test handling of floating point coordinates"""
    node = BBoxToMask()

    # Bbox with float coordinates (should be converted to int)
    bbox = [[10.7, 20.3, 50.9, 50.1]]

    result = node.bbox_to_mask(bbox, width=256, height=256)
    mask = result[0]

    # Should work - floats converted to ints
    # int(10.7)=10, int(20.3)=20, int(50.9)=50, int(50.1)=50
    assert mask[20:70, 10:60].sum() == 50 * 50, "Should handle float coordinates"

    print("✓ test_float_coordinates passed")


def test_return_types():
    """Validate return types match RETURN_TYPES"""
    node = BBoxToMask()

    bbox = [[10, 10, 50, 50]]
    result = node.bbox_to_mask(bbox, 256, 256)

    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 1, "Should return 1 item (matching RETURN_TYPES)"

    mask = result[0]
    assert isinstance(mask, torch.Tensor), "Mask should be tensor"
    assert mask.dtype == torch.float32, "Mask should be float32"

    print("✓ test_return_types passed")


def test_input_types_structure():
    """Validate INPUT_TYPES matches function signature"""
    input_types = BBoxToMask.INPUT_TYPES()

    all_inputs = set()
    if "required" in input_types:
        all_inputs.update(input_types["required"].keys())
    if "optional" in input_types:
        all_inputs.update(input_types["optional"].keys())

    function = getattr(BBoxToMask(), BBoxToMask.FUNCTION)
    sig = inspect.signature(function)
    function_params = set(sig.parameters.keys()) - {'self'}

    missing = function_params - all_inputs
    extra = all_inputs - function_params

    assert not missing, f"Function has params not in INPUT_TYPES: {missing}"
    assert not extra, f"INPUT_TYPES has entries not in function: {extra}"

    print("✓ test_input_types_structure passed")


def test_output_is_not_list():
    """Validate that OUTPUT_IS_LIST is not set (removed in refactor)"""
    # Node should not have OUTPUT_IS_LIST, or it should be False/empty
    has_output_is_list = hasattr(BBoxToMask, 'OUTPUT_IS_LIST')

    if has_output_is_list:
        output_is_list = BBoxToMask.OUTPUT_IS_LIST
        # If it exists, make sure it's False or empty
        if isinstance(output_is_list, (list, tuple)):
            assert all(not x for x in output_is_list), "OUTPUT_IS_LIST should be all False"
        else:
            assert not output_is_list, "OUTPUT_IS_LIST should be False"

    # Either way, return should be single mask (not list)
    assert BBoxToMask.RETURN_TYPES == ("MASK",), "Should return single MASK type"
    assert BBoxToMask.RETURN_NAMES == ("mask",), "Should return single 'mask' output"

    print("✓ test_output_is_not_list passed")


def run_all_tests():
    """Run all test functions"""
    print("Running tests for BBoxToMask (simplified single bbox version)...\n")

    try:
        test_single_bbox()
        test_wrapped_bbox_format()
        test_unwrapped_bbox_format()
        test_bbox_clamping()
        test_bbox_completely_outside()
        test_invert_mode()
        test_zero_size_bbox()
        test_empty_bbox()
        test_invalid_bbox()
        test_different_image_sizes()
        test_float_coordinates()
        test_return_types()
        test_input_types_structure()
        test_output_is_not_list()

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
