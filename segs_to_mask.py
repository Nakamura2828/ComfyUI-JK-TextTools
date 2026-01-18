"""
SEGs to Mask Node for ComfyUI

Converts SEGS (segmentation results) to binary masks.
SEGS format: ((height, width), [SEG(...), SEG(...), ...])
Each SEG contains a cropped_mask that gets placed back into the full image.
"""

import torch
import numpy as np
import fnmatch


class SEGsToMask:
    """
    Convert SEGS (segmentation results) to masks.

    Takes SEGS tuple and creates:
    - Combined mask (union of all segments)
    - Individual masks for each segment (as list)

    SEGS format from SAM3 and similar nodes:
    ((height, width), [SEG(...), SEG(...), ...])

    Each SEG object has:
    - cropped_mask: numpy array with mask data
    - crop_region: [x1, y1, x2, y2] where mask should be placed
    - label: string label (e.g., "person_0")
    - confidence: float confidence score
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS", {}),
            },
            "optional": {
                "label_filter": ("STRING", {
                    "default": "*",
                    "multiline": False
                }),
                "min_confidence": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "min_area_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "sort_order": (["default", "x_then_y", "y_then_x", "confidence_high_to_low"], {
                    "default": "default"
                }),
                "union_same_labels": ("BOOLEAN", {
                    "default": True
                }),
                "invert": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "STRING", "INT")
    RETURN_NAMES = ("combined_mask", "individual_masks", "labels_info", "seg_count")
    OUTPUT_IS_LIST = (False, True, True, False)
    FUNCTION = "segs_to_mask"
    CATEGORY = "JK-TextTools/segs"

    def segs_to_mask(self, segs, label_filter="*", min_confidence=0.0, min_area_percent=0.0, sort_order="default", union_same_labels=True, invert=False):
        """
        Convert SEGS to masks.

        Args:
            segs: SEGS tuple ((height, width), [SEG(...), ...])
            label_filter: Wildcard pattern for filtering by label (*, ?, [abc])
            min_confidence: Minimum confidence threshold (0.0-1.0)
            min_area_percent: Minimum mask area as percentage of image (0.0-100.0)
            sort_order: Order segments ("default", "x_then_y", "y_then_x")
            union_same_labels: If True, combine all segments with same label into one mask
            invert: If True, mask areas are 0, rest is 1

        Returns:
            tuple: (combined_mask, list_of_individual_masks, list_of_labels_info, count)
        """
        # Validate SEGS format
        if not isinstance(segs, tuple) or len(segs) != 2:
            # Invalid SEGS format - return empty
            empty_mask = torch.zeros((512, 512), dtype=torch.float32)
            return (empty_mask, [empty_mask], [""], 0)

        dims, seg_list = segs

        # Extract dimensions
        if isinstance(dims, tuple) and len(dims) == 2:
            height, width = dims
        else:
            # Invalid dimensions - return empty
            empty_mask = torch.zeros((512, 512), dtype=torch.float32)
            return (empty_mask, [empty_mask], [""], 0)

        # Validate seg_list
        if not isinstance(seg_list, list) or len(seg_list) == 0:
            # Empty seg list - return empty masks
            empty_mask = torch.zeros((height, width), dtype=torch.float32)
            return (empty_mask, [empty_mask], [""], 0)

        # Sort segments if requested
        if sort_order != "default":
            seg_list = self._sort_segments(seg_list, sort_order)

        # If union_same_labels, group segments by label first
        if union_same_labels:
            seg_groups = self._group_segments_by_label(seg_list)
        else:
            # Treat each segment as its own group
            seg_groups = {f"_seg_{i}": [seg] for i, seg in enumerate(seg_list)}

        # Create combined mask
        combined_mask = torch.zeros((height, width), dtype=torch.float32)
        individual_masks = []
        labels_info = []
        image_area = height * width

        # Process each group (either by label or individual segments)
        for label, group_segs in seg_groups.items():
            # For union mode, track max confidence and union mask for this label
            group_mask = torch.zeros((height, width), dtype=torch.float32)
            max_confidence = 0.0
            group_has_valid_mask = False

            for seg in group_segs:
                # Extract SEG attributes
                try:
                    cropped_mask = getattr(seg, 'cropped_mask', None)
                    crop_region = getattr(seg, 'crop_region', None)
                    seg_label = getattr(seg, 'label', '')
                    confidence = getattr(seg, 'confidence', 1.0)

                    # Handle confidence as numpy array (ImpactPack) or float (TBG SAM3)
                    if isinstance(confidence, np.ndarray):
                        confidence = float(confidence.item())
                    elif not isinstance(confidence, (int, float)):
                        confidence = float(confidence)
                except Exception:
                    # If SEG is not an object, skip it
                    continue

                # Apply filters
                if not self._matches_filter(seg_label, label_filter):
                    continue

                if confidence < min_confidence:
                    continue

                # Track max confidence for this group
                max_confidence = max(max_confidence, confidence)

                # Handle missing cropped_mask - skip but keep tracking confidence
                if cropped_mask is None:
                    continue

                # Validate crop_region
                if crop_region is None or len(crop_region) != 4:
                    continue

                # Convert cropped_mask to tensor if it's numpy
                if isinstance(cropped_mask, np.ndarray):
                    mask_tensor = torch.from_numpy(cropped_mask).float()
                else:
                    mask_tensor = cropped_mask

                # Create full-size mask for this segment piece
                segment_mask = torch.zeros((height, width), dtype=torch.float32)

                # Extract crop region coordinates
                x1, y1, x2, y2 = crop_region
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Clamp coordinates to image bounds
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                # Calculate actual region size
                region_h = y2 - y1
                region_w = x2 - x1

                if region_h <= 0 or region_w <= 0:
                    continue

                # Validate mask_tensor shape before unpacking
                if mask_tensor.ndim != 2:
                    # Invalid mask dimensions - skip this segment
                    continue

                # Resize mask_tensor to fit crop region if needed
                mask_h, mask_w = mask_tensor.shape

                if mask_h != region_h or mask_w != region_w:
                    # Crop or pad mask to match region size
                    copy_h = min(mask_h, region_h)
                    copy_w = min(mask_w, region_w)
                    segment_mask[y1:y1+copy_h, x1:x1+copy_w] = mask_tensor[:copy_h, :copy_w]
                else:
                    # Perfect fit - place mask directly
                    segment_mask[y1:y2, x1:x2] = mask_tensor

                # Union this segment piece into the group mask
                group_mask = torch.max(group_mask, segment_mask)
                group_has_valid_mask = True

            # After processing all segments in this group, check if we have a valid mask
            if not group_has_valid_mask:
                continue

            # Check minimum area filter on the combined group mask
            mask_area = group_mask.sum().item()
            area_percentage = (mask_area / image_area) * 100

            if area_percentage < min_area_percent:
                # Group mask too small - skip it
                continue

            # Add group mask to combined mask (union)
            combined_mask = torch.max(combined_mask, group_mask)

            # Add to individual masks list
            individual_masks.append(group_mask)

            # Use the actual label (extract from group key if needed)
            display_label = label if union_same_labels else seg_label
            labels_info.append(f"{display_label}: {max_confidence:.2f}")

        # Apply invert if needed
        if invert:
            combined_mask = 1.0 - combined_mask
            individual_masks = [1.0 - mask for mask in individual_masks]

        # If no valid segments after filtering, return empty
        if len(individual_masks) == 0:
            empty_mask = torch.zeros((height, width), dtype=torch.float32)
            individual_masks = [empty_mask]
            labels_info = [""]

        count = len(individual_masks)

        return (combined_mask, individual_masks, labels_info, count)

    def _matches_filter(self, label, pattern):
        """
        Check if label matches wildcard pattern.

        Args:
            label: Label string to check
            pattern: Wildcard pattern (*, ?, [abc])

        Returns:
            bool: True if matches
        """
        if pattern == "*":
            return True

        if not label:
            return pattern == ""

        return fnmatch.fnmatch(label, pattern)

    def _group_segments_by_label(self, seg_list):
        """
        Group segments by their label.

        Args:
            seg_list: List of SEG objects

        Returns:
            Dictionary mapping label -> list of SEG objects with that label
        """
        groups = {}
        for seg in seg_list:
            try:
                label = getattr(seg, 'label', '')
            except Exception:
                label = ''

            if label not in groups:
                groups[label] = []
            groups[label].append(seg)

        return groups

    def _sort_segments(self, seg_list, sort_order):
        """
        Sort segments by position or confidence.

        Args:
            seg_list: List of SEG objects
            sort_order: "x_then_y", "y_then_x", or "confidence_high_to_low"

        Returns:
            Sorted list of SEG objects
        """
        if sort_order == "confidence_high_to_low":
            # Sort by confidence (highest first)
            def get_confidence(seg):
                try:
                    confidence = getattr(seg, 'confidence', 0.0)
                    # Handle numpy array confidence (ImpactPack)
                    if isinstance(confidence, np.ndarray):
                        confidence = float(confidence.item())
                    elif not isinstance(confidence, (int, float)):
                        confidence = float(confidence)
                    return -confidence  # Negative for descending sort
                except Exception:
                    return 0.0

            return sorted(seg_list, key=get_confidence)

        # Position-based sorting
        def get_sort_key(seg):
            # Try to get coordinates from crop_region first, fallback to bbox
            crop_region = getattr(seg, 'crop_region', None)
            if crop_region and len(crop_region) >= 4:
                x1, y1, x2, y2 = crop_region[:4]
                x = int(x1)
                y = int(y1)
            else:
                # Fallback to bbox
                bbox = getattr(seg, 'bbox', None)
                if bbox and len(bbox) >= 2:
                    x, y = int(bbox[0]), int(bbox[1])
                else:
                    # No valid coordinates, put at end
                    x, y = float('inf'), float('inf')

            if sort_order == "x_then_y":
                return (x, y)  # Sort by x first, then y
            else:  # y_then_x
                return (y, x)  # Sort by y first, then x

        return sorted(seg_list, key=get_sort_key)
