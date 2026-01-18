"""
Tests for SEGs to Mask Node

Tests the SEGS (segmentation) to mask converter with filtering and union features.

Run from project root:
    python tests/test_segs_to_mask.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import node modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import inspect
from segs_to_mask import SEGsToMask


# Mock SEG class for testing
class MockSEG:
    """Mock SEG object for testing"""
    def __init__(self, cropped_mask, crop_region, label, confidence):
        self.cropped_mask = cropped_mask
        self.crop_region = crop_region
        self.label = label
        self.confidence = confidence
        self.bbox = (crop_region[0], crop_region[1],
                     crop_region[2] - crop_region[0],
                     crop_region[3] - crop_region[1])


def create_mock_segs(height=512, width=512, seg_data=None):
    """
    Create mock SEGS tuple for testing.

    Args:
        height: Image height
        width: Image width
        seg_data: List of tuples (mask_array, crop_region, label, confidence)

    Returns:
        SEGS tuple: ((height, width), [SEG(...), ...])
    """
    if seg_data is None:
        seg_data = []

    seg_list = []
    for mask, region, label, conf in seg_data:
        # Convert mask to numpy if it's not already
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask, dtype=np.float32)
        seg_list.append(MockSEG(mask, region, label, conf))

    return ((height, width), seg_list)


def test_basic_single_seg():
    """Test converting single SEG to mask"""
    node = SEGsToMask()

    # Create a simple 50x50 mask at position (10, 10)
    mask = np.ones((50, 50), dtype=np.float32)
    segs = create_mock_segs(512, 512, [
        (mask, [10, 10, 60, 60], "person_0", 0.95)
    ])

    combined, individual, labels, count = node.segs_to_mask(segs)

    # Check return structure
    assert isinstance(combined, torch.Tensor), "Combined mask should be tensor"
    assert combined.shape == (512, 512), f"Combined mask shape: {combined.shape}"
    assert len(individual) == 1, f"Should have 1 individual mask, got {len(individual)}"
    assert len(labels) == 1, f"Should have 1 label, got {len(labels)}"
    assert count == 1, f"Count should be 1, got {count}"

    # Check mask content
    assert combined[10:60, 10:60].sum() == 50 * 50, "Mask should fill crop region"
    assert labels[0] == "person_0: 0.95", f"Label should be 'person_0: 0.95', got {labels[0]}"

    print("✓ test_basic_single_seg passed")


def test_union_same_labels():
    """Test union of segments with same label"""
    node = SEGsToMask()

    # Create two masks for same label at different positions
    mask1 = np.ones((30, 30), dtype=np.float32)
    mask2 = np.ones((20, 20), dtype=np.float32)

    segs = create_mock_segs(512, 512, [
        (mask1, [10, 10, 40, 40], "person_0", 0.95),
        (mask2, [100, 100, 120, 120], "person_0", 0.90)
    ])

    combined, individual, labels, count = node.segs_to_mask(segs, union_same_labels=True)

    # Should have only 1 output mask (union of both)
    assert count == 1, f"Should have 1 unified mask, got {count}"
    assert len(individual) == 1, f"Should have 1 individual mask, got {len(individual)}"
    assert len(labels) == 1, f"Should have 1 label, got {len(labels)}"

    # Check that both regions are filled in the unified mask
    unified_mask = individual[0]
    assert unified_mask[10:40, 10:40].sum() == 30 * 30, "First region should be filled"
    assert unified_mask[100:120, 100:120].sum() == 20 * 20, "Second region should be filled"

    # Should use max confidence (0.95)
    assert labels[0] == "person_0: 0.95", f"Should use max confidence, got {labels[0]}"

    print("✓ test_union_same_labels passed")


def test_no_union_separate_masks():
    """Test without union - each segment gets own mask"""
    node = SEGsToMask()

    mask1 = np.ones((30, 30), dtype=np.float32)
    mask2 = np.ones((20, 20), dtype=np.float32)

    segs = create_mock_segs(512, 512, [
        (mask1, [10, 10, 40, 40], "person_0", 0.95),
        (mask2, [100, 100, 120, 120], "person_0", 0.90)
    ])

    combined, individual, labels, count = node.segs_to_mask(segs, union_same_labels=False)

    # Should have 2 separate masks
    assert count == 2, f"Should have 2 separate masks, got {count}"
    assert len(individual) == 2, f"Should have 2 individual masks, got {len(individual)}"
    assert len(labels) == 2, f"Should have 2 labels, got {len(labels)}"

    # Each mask should only have one region
    mask1 = individual[0]
    mask2 = individual[1]
    assert mask1[10:40, 10:40].sum() == 30 * 30, "First mask should have first region"
    assert mask1[100:120, 100:120].sum() == 0, "First mask should not have second region"
    assert mask2[100:120, 100:120].sum() == 20 * 20, "Second mask should have second region"
    assert mask2[10:40, 10:40].sum() == 0, "Second mask should not have first region"

    print("✓ test_no_union_separate_masks passed")


def test_multiple_labels():
    """Test with multiple different labels"""
    node = SEGsToMask()

    mask1 = np.ones((30, 30), dtype=np.float32)
    mask2 = np.ones((20, 20), dtype=np.float32)
    mask3 = np.ones((25, 25), dtype=np.float32)

    segs = create_mock_segs(512, 512, [
        (mask1, [10, 10, 40, 40], "person_0", 0.95),
        (mask2, [100, 100, 120, 120], "person_1", 0.90),
        (mask3, [200, 200, 225, 225], "person_2", 0.85)
    ])

    combined, individual, labels, count = node.segs_to_mask(segs, union_same_labels=True)

    # Should have 3 masks (one per unique label)
    assert count == 3, f"Should have 3 masks, got {count}"
    assert len(set(l.split(":")[0] for l in labels)) == 3, "Should have 3 unique labels"

    print("✓ test_multiple_labels passed")


def test_label_filter_wildcard():
    """Test label filtering with wildcards"""
    node = SEGsToMask()

    mask1 = np.ones((30, 30), dtype=np.float32)
    mask2 = np.ones((20, 20), dtype=np.float32)
    mask3 = np.ones((25, 25), dtype=np.float32)

    segs = create_mock_segs(512, 512, [
        (mask1, [10, 10, 40, 40], "person_0", 0.95),
        (mask2, [100, 100, 120, 120], "person_1", 0.90),
        (mask3, [200, 200, 225, 225], "dog_0", 0.85)
    ])

    # Filter for only "person_*"
    combined, individual, labels, count = node.segs_to_mask(segs, label_filter="person_*")

    # Should only have person masks
    assert count == 2, f"Should have 2 person masks, got {count}"
    assert all("person" in l for l in labels), f"All labels should contain 'person', got {labels}"

    print("✓ test_label_filter_wildcard passed")


def test_confidence_filter():
    """Test filtering by minimum confidence"""
    node = SEGsToMask()

    mask1 = np.ones((30, 30), dtype=np.float32)
    mask2 = np.ones((20, 20), dtype=np.float32)
    mask3 = np.ones((25, 25), dtype=np.float32)

    segs = create_mock_segs(512, 512, [
        (mask1, [10, 10, 40, 40], "person_0", 0.95),
        (mask2, [100, 100, 120, 120], "person_1", 0.80),
        (mask3, [200, 200, 225, 225], "person_2", 0.70)
    ])

    # Filter for confidence >= 0.85
    combined, individual, labels, count = node.segs_to_mask(segs, min_confidence=0.85)

    # Should only have high-confidence mask
    assert count == 1, f"Should have 1 high-confidence mask, got {count}"
    assert "0.95" in labels[0], f"Should have 0.95 confidence, got {labels[0]}"

    print("✓ test_confidence_filter passed")


def test_min_area_filter():
    """Test filtering by minimum area percentage"""
    node = SEGsToMask()

    # Create masks of different sizes
    large_mask = np.ones((100, 100), dtype=np.float32)  # 10000 pixels
    small_mask = np.ones((10, 10), dtype=np.float32)    # 100 pixels

    # At 512x512 = 262144 pixels:
    # - 10000 pixels = 3.8%
    # - 100 pixels = 0.038%

    segs = create_mock_segs(512, 512, [
        (large_mask, [10, 10, 110, 110], "large", 0.95),
        (small_mask, [200, 200, 210, 210], "small", 0.95)
    ])

    # Filter for area >= 1%
    combined, individual, labels, count = node.segs_to_mask(segs, min_area_percent=1.0)

    # Should only have large mask
    assert count == 1, f"Should have 1 large mask, got {count}"
    assert "large" in labels[0], f"Should have 'large' label, got {labels[0]}"

    print("✓ test_min_area_filter passed")


def test_sort_order_x_then_y():
    """Test sorting by x coordinate then y"""
    node = SEGsToMask()

    mask = np.ones((20, 20), dtype=np.float32)

    segs = create_mock_segs(512, 512, [
        (mask, [100, 10, 120, 30], "seg_0", 0.95),   # x=100, y=10
        (mask, [10, 100, 30, 120], "seg_1", 0.95),   # x=10, y=100
        (mask, [100, 100, 120, 120], "seg_2", 0.95), # x=100, y=100
    ])

    combined, individual, labels, count = node.segs_to_mask(
        segs, sort_order="x_then_y", union_same_labels=False
    )

    # Should be sorted: seg_1 (x=10), seg_0 (x=100, y=10), seg_2 (x=100, y=100)
    assert "seg_1" in labels[0], f"First should be seg_1, got {labels[0]}"
    assert "seg_0" in labels[1], f"Second should be seg_0, got {labels[1]}"
    assert "seg_2" in labels[2], f"Third should be seg_2, got {labels[2]}"

    print("✓ test_sort_order_x_then_y passed")


def test_sort_order_confidence():
    """Test sorting by confidence (high to low)"""
    node = SEGsToMask()

    mask = np.ones((20, 20), dtype=np.float32)

    segs = create_mock_segs(512, 512, [
        (mask, [10, 10, 30, 30], "seg_0", 0.75),
        (mask, [50, 50, 70, 70], "seg_1", 0.95),
        (mask, [100, 100, 120, 120], "seg_2", 0.60),
    ])

    combined, individual, labels, count = node.segs_to_mask(
        segs, sort_order="confidence_high_to_low", union_same_labels=False
    )

    # Should be sorted: seg_1 (0.95), seg_0 (0.75), seg_2 (0.60)
    assert "seg_1: 0.95" in labels[0], f"First should be seg_1 with 0.95, got {labels[0]}"
    assert "seg_0: 0.75" in labels[1], f"Second should be seg_0 with 0.75, got {labels[1]}"
    assert "seg_2: 0.60" in labels[2], f"Third should be seg_2 with 0.60, got {labels[2]}"

    print("✓ test_sort_order_confidence passed")


def test_sort_order_y_then_x():
    """Test sorting by y coordinate then x"""
    node = SEGsToMask()

    mask = np.ones((20, 20), dtype=np.float32)

    segs = create_mock_segs(512, 512, [
        (mask, [100, 10, 120, 30], "seg_0", 0.95),   # x=100, y=10
        (mask, [10, 100, 30, 120], "seg_1", 0.95),   # x=10, y=100
        (mask, [100, 100, 120, 120], "seg_2", 0.95), # x=100, y=100
    ])

    combined, individual, labels, count = node.segs_to_mask(
        segs, sort_order="y_then_x", union_same_labels=False
    )

    # Should be sorted: seg_0 (y=10), seg_1 (y=100, x=10), seg_2 (y=100, x=100)
    assert "seg_0" in labels[0], f"First should be seg_0, got {labels[0]}"
    assert "seg_1" in labels[1], f"Second should be seg_1, got {labels[1]}"
    assert "seg_2" in labels[2], f"Third should be seg_2, got {labels[2]}"

    print("✓ test_sort_order_y_then_x passed")


def test_invert_mode():
    """Test inverted masks"""
    node = SEGsToMask()

    mask = np.ones((50, 50), dtype=np.float32)
    segs = create_mock_segs(512, 512, [
        (mask, [10, 10, 60, 60], "person_0", 0.95)
    ])

    combined, individual, labels, count = node.segs_to_mask(segs, invert=True)

    # Mask region should be 0, rest should be 1
    assert combined[10:60, 10:60].sum() == 0, "Mask region should be 0 when inverted"
    assert combined[0:10, 0:10].sum() > 0, "Outside region should be 1 when inverted"

    print("✓ test_invert_mode passed")


def test_empty_segs():
    """Test with empty SEGS list"""
    node = SEGsToMask()

    segs = ((512, 512), [])

    combined, individual, labels, count = node.segs_to_mask(segs)

    assert count == 0, f"Count should be 0, got {count}"
    assert len(individual) == 1, "Should return one empty mask"
    assert individual[0].sum() == 0, "Empty mask should be all zeros"

    print("✓ test_empty_segs passed")


def test_none_cropped_mask():
    """Test handling of None cropped_mask"""
    node = SEGsToMask()

    # Valid mask and None mask
    mask = np.ones((30, 30), dtype=np.float32)
    segs = create_mock_segs(512, 512, [
        (mask, [10, 10, 40, 40], "person_0", 0.95),
        (None, [100, 100, 130, 130], "person_1", 0.90)
    ])

    combined, individual, labels, count = node.segs_to_mask(segs)

    # Should only process valid mask
    assert count == 1, f"Should have 1 valid mask, got {count}"
    assert "person_0" in labels[0], f"Should have person_0, got {labels[0]}"

    print("✓ test_none_cropped_mask passed")


def test_crop_region_clamping():
    """Test that crop regions are clamped to image bounds"""
    node = SEGsToMask()

    # Mask that extends beyond image bounds
    mask = np.ones((100, 100), dtype=np.float32)
    segs = create_mock_segs(512, 512, [
        (mask, [480, 480, 580, 580], "person_0", 0.95)  # Extends beyond 512x512
    ])

    combined, individual, labels, count = node.segs_to_mask(segs)

    # Should clamp to image bounds
    assert count == 1, "Should successfully process clamped region"
    # Should fill from 480 to 512 (32x32 region)
    assert combined[480:512, 480:512].sum() == 32 * 32, "Should fill clamped region"

    print("✓ test_crop_region_clamping passed")


def test_combined_mask_union():
    """Test that combined mask is proper union of all individual masks"""
    node = SEGsToMask()

    mask1 = np.ones((30, 30), dtype=np.float32)
    mask2 = np.ones((20, 20), dtype=np.float32)

    segs = create_mock_segs(512, 512, [
        (mask1, [10, 10, 40, 40], "person_0", 0.95),
        (mask2, [100, 100, 120, 120], "person_1", 0.90)
    ])

    combined, individual, labels, count = node.segs_to_mask(segs)

    # Combined should have both regions
    assert combined[10:40, 10:40].sum() == 30 * 30, "Combined should have first region"
    assert combined[100:120, 100:120].sum() == 20 * 20, "Combined should have second region"

    # Combined should equal union of individuals
    manual_union = torch.zeros((512, 512), dtype=torch.float32)
    for mask in individual:
        manual_union = torch.max(manual_union, mask)

    assert torch.equal(combined, manual_union), "Combined should equal union of individuals"

    print("✓ test_combined_mask_union passed")


def test_input_types_structure():
    """Validate INPUT_TYPES matches function signature"""
    input_types = SEGsToMask.INPUT_TYPES()

    all_inputs = set()
    if "required" in input_types:
        all_inputs.update(input_types["required"].keys())
    if "optional" in input_types:
        all_inputs.update(input_types["optional"].keys())

    function = getattr(SEGsToMask(), SEGsToMask.FUNCTION)
    sig = inspect.signature(function)
    function_params = set(sig.parameters.keys()) - {'self'}

    missing = function_params - all_inputs
    extra = all_inputs - function_params

    assert not missing, f"Function has params not in INPUT_TYPES: {missing}"
    assert not extra, f"INPUT_TYPES has entries not in function: {extra}"

    print("✓ test_input_types_structure passed")


def test_return_types():
    """Validate return types match RETURN_TYPES"""
    node = SEGsToMask()

    mask = np.ones((30, 30), dtype=np.float32)
    segs = create_mock_segs(512, 512, [
        (mask, [10, 10, 40, 40], "person_0", 0.95)
    ])

    result = node.segs_to_mask(segs)

    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 4, f"Should return 4 items, got {len(result)}"

    combined, individual, labels, count = result

    assert isinstance(combined, torch.Tensor), "Combined mask should be tensor"
    assert isinstance(individual, list), "Individual masks should be list"
    assert isinstance(labels, list), "Labels should be list"
    assert isinstance(count, int), "Count should be int"

    assert all(isinstance(m, torch.Tensor) for m in individual), "All individual masks should be tensors"
    assert all(isinstance(l, str) for l in labels), "All labels should be strings"

    print("✓ test_return_types passed")


def test_numpy_array_confidence():
    """Test handling of numpy array confidence values (ImpactPack format)"""
    node = SEGsToMask()

    # Create SEG with numpy array confidence (like ImpactPack)
    mask = np.ones((50, 50), dtype=np.float32)
    seg_data = [
        (mask, [10, 10, 60, 60], "person", np.array([0.9457324], dtype=np.float32)),
        (mask, [100, 100, 150, 150], "dog", np.array([0.8123456], dtype=np.float32))
    ]

    segs = create_mock_segs(256, 256, seg_data)

    combined_mask, individual_masks, labels_info, count = node.segs_to_mask(segs)

    assert count == 2, f"Should have 2 masks, got {count}"
    assert len(labels_info) == 2, "Should have 2 labels"

    # Check that confidence values are properly formatted as floats
    assert "person: 0.95" in labels_info[0], f"Expected 'person: 0.95', got '{labels_info[0]}'"
    assert "dog: 0.81" in labels_info[1], f"Expected 'dog: 0.81', got '{labels_info[1]}'"

    print("✓ test_numpy_array_confidence passed")


def run_all_tests():
    """Run all test functions"""
    print("Running tests for SEGsToMask...\n")

    try:
        test_basic_single_seg()
        test_union_same_labels()
        test_no_union_separate_masks()
        test_multiple_labels()
        test_label_filter_wildcard()
        test_confidence_filter()
        test_min_area_filter()
        test_sort_order_x_then_y()
        test_sort_order_confidence()
        test_sort_order_y_then_x()
        test_invert_mode()
        test_empty_segs()
        test_none_cropped_mask()
        test_crop_region_clamping()
        test_combined_mask_union()
        test_input_types_structure()
        test_return_types()
        test_numpy_array_confidence()

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
