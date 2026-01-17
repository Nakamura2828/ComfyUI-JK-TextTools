"""
Tests for BBox to Mask Node

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
    
    individual_masks, combined_mask = node.bbox_to_mask(bbox, width=512, height=512)
    
    # Check individual masks
    assert len(individual_masks) == 1, "Should have 1 mask"
    mask = individual_masks[0]
    assert mask.shape == (512, 512), f"Mask should be 512x512, got {mask.shape}"
    assert isinstance(mask, torch.Tensor), "Mask should be tensor"
    
    # Check bbox area is 1, rest is 0
    assert mask[20:120, 10:110].sum() == 100 * 100, "Bbox area should be all 1s"
    assert mask[0:20, :].sum() == 0, "Area above bbox should be all 0s"
    assert mask[:, 0:10].sum() == 0, "Area left of bbox should be all 0s"
    
    # Combined should be same as individual for single bbox
    assert torch.equal(combined_mask, mask), "Combined should equal individual for single bbox"
    
    print("✓ test_single_bbox passed")


def test_multiple_bboxes():
    """Test converting multiple bboxes to masks"""
    node = BBoxToMask()
    
    # Two non-overlapping bboxes
    bbox = [[10, 10, 50, 50], [100, 100, 50, 50]]
    
    individual_masks, combined_mask = node.bbox_to_mask(bbox, width=256, height=256)
    
    # Should have 2 individual masks
    assert len(individual_masks) == 2, "Should have 2 masks"
    
    # First mask should only have first bbox
    mask1 = individual_masks[0]
    assert mask1[10:60, 10:60].sum() == 50 * 50, "First bbox area"
    assert mask1[100:150, 100:150].sum() == 0, "Should not have second bbox"
    
    # Second mask should only have second bbox
    mask2 = individual_masks[1]
    assert mask2[100:150, 100:150].sum() == 50 * 50, "Second bbox area"
    assert mask2[10:60, 10:60].sum() == 0, "Should not have first bbox"
    
    # Combined should have both
    assert combined_mask[10:60, 10:60].sum() == 50 * 50, "Combined should have first bbox"
    assert combined_mask[100:150, 100:150].sum() == 50 * 50, "Combined should have second bbox"
    
    print("✓ test_multiple_bboxes passed")


def test_overlapping_bboxes():
    """Test overlapping bboxes union correctly"""
    node = BBoxToMask()
    
    # Two overlapping bboxes
    bbox = [[10, 10, 60, 60], [40, 40, 60, 60]]
    
    individual_masks, combined_mask = node.bbox_to_mask(bbox, width=256, height=256)
    
    # Combined mask should be union (not additive)
    # Each pixel should be max 1.0, not 2.0
    assert combined_mask.max() <= 1.0, "Mask values should not exceed 1.0"
    
    # Overlapping region should be in combined
    assert combined_mask[50, 50] == 1.0, "Overlapping area should be 1"
    
    print("✓ test_overlapping_bboxes passed")


def test_bbox_clamping():
    """Test that bboxes outside image bounds are clamped"""
    node = BBoxToMask()
    
    # Bbox that extends beyond image
    bbox = [[-10, -10, 100, 100]]  # Partially outside
    
    individual_masks, combined_mask = node.bbox_to_mask(bbox, width=256, height=256)
    
    # Should only fill the part that's inside
    mask = individual_masks[0]
    # Should fill from (0,0) to (90,90) - the part that's inside
    assert mask[0:90, 0:90].sum() > 0, "Should fill inside portion"
    assert mask[0, 0] == 1.0, "Top-left should be filled"
    
    print("✓ test_bbox_clamping passed")


def test_invert_mode():
    """Test inverted mask (bbox is black, rest is white)"""
    node = BBoxToMask()
    
    bbox = [[10, 10, 50, 50]]
    
    individual_masks, combined_mask = node.bbox_to_mask(bbox, width=256, height=256, invert=True)
    
    mask = individual_masks[0]
    
    # Bbox area should be 0 (black)
    assert mask[10:60, 10:60].sum() == 0, "Bbox area should be 0 when inverted"
    
    # Rest should be 1 (white)
    assert mask[0:10, 0:10].sum() > 0, "Area outside bbox should be 1"
    assert mask[0, 0] == 1.0, "Outside area should be 1"
    
    print("✓ test_invert_mode passed")


def test_zero_size_bbox():
    """Test handling of zero or negative size bbox"""
    node = BBoxToMask()
    
    # Zero width bbox
    bbox = [[10, 10, 0, 50]]
    
    individual_masks, combined_mask = node.bbox_to_mask(bbox, width=256, height=256)
    
    # Should produce empty mask
    assert combined_mask.sum() == 0, "Zero-size bbox should produce empty mask"
    
    print("✓ test_zero_size_bbox passed")


def test_empty_bbox():
    """Test handling of empty bbox list"""
    node = BBoxToMask()
    
    bbox = []
    
    individual_masks, combined_mask = node.bbox_to_mask(bbox, width=256, height=256)
    
    # Should return empty mask
    assert len(individual_masks) == 1, "Should return at least one mask"
    assert combined_mask.sum() == 0, "Empty bbox should produce empty mask"
    
    print("✓ test_empty_bbox passed")


def test_different_image_sizes():
    """Test with various image dimensions"""
    node = BBoxToMask()
    
    # Small image
    bbox = [[5, 5, 10, 10]]
    individual_masks, _ = node.bbox_to_mask(bbox, width=64, height=64)
    assert individual_masks[0].shape == (64, 64)
    
    # Large image
    individual_masks, _ = node.bbox_to_mask(bbox, width=2048, height=2048)
    assert individual_masks[0].shape == (2048, 2048)
    
    # Non-square
    individual_masks, _ = node.bbox_to_mask(bbox, width=1920, height=1080)
    assert individual_masks[0].shape == (1080, 1920)
    
    print("✓ test_different_image_sizes passed")


def test_return_types():
    """Validate return types"""
    node = BBoxToMask()
    
    bbox = [[10, 10, 50, 50]]
    result = node.bbox_to_mask(bbox, 256, 256)
    
    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 2, "Should return 2 items"
    
    individual_masks, combined_mask = result
    
    assert isinstance(individual_masks, list), "individual_masks should be list"
    assert isinstance(individual_masks[0], torch.Tensor), "Each mask should be tensor"
    assert isinstance(combined_mask, torch.Tensor), "combined_mask should be tensor"
    
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


def run_all_tests():
    """Run all test functions"""
    print("Running tests for BBoxToMask...\n")
    
    try:
        test_single_bbox()
        test_multiple_bboxes()
        test_overlapping_bboxes()
        test_bbox_clamping()
        test_invert_mode()
        test_zero_size_bbox()
        test_empty_bbox()
        test_different_image_sizes()
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
