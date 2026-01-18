"""
SEGs to SAM3 Query Node for ComfyUI

Converts SEGS segmentation to TBG SAM3 Selector query formats.
Generates box query (bounding box) and point query (mask centroid).
"""

import json
import torch
import numpy as np


class SEGsToSAM3Query:
    """
    Convert SEGS segmentation to SAM3 Selector query formats.

    Takes SEGS and generates:
    - Box query (XYXY format): Bounding box of entire segmentation
    - Point query: Centroid of mask (center of mass)

    SEGS format: ((height, width), [SEG(...), SEG(...), ...])
    Box query: [{"x1": float, "y1": float, "x2": float, "y2": float}]
    Point query: [{"x": float, "y": float}]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS", {}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("box_query", "point_query")
    FUNCTION = "segs_to_sam3_query"
    CATEGORY = "JK-TextTools/segs"

    def segs_to_sam3_query(self, segs):
        """
        Convert SEGS to SAM3 query formats.

        Args:
            segs: SEGS tuple ((height, width), [SEG(...), ...])

        Returns:
            tuple: (box_query_string, point_query_string)
        """
        # Validate SEGS format
        if not isinstance(segs, tuple) or len(segs) != 2:
            # Invalid SEGS format - return empty queries
            return ("[]", "[]")

        dims, seg_list = segs

        # Extract dimensions
        if isinstance(dims, tuple) and len(dims) == 2:
            height, width = dims
        else:
            # Invalid dimensions - return empty
            return ("[]", "[]")

        # Validate seg_list
        if not isinstance(seg_list, list) or len(seg_list) == 0:
            # Empty seg list - return empty queries
            return ("[]", "[]")

        # Reconstruct full mask by unioning all segments
        full_mask = torch.zeros((height, width), dtype=torch.float32)
        has_valid_mask = False

        for seg in seg_list:
            # Extract SEG attributes
            try:
                cropped_mask = getattr(seg, 'cropped_mask', None)
                crop_region = getattr(seg, 'crop_region', None)
            except Exception:
                # If SEG is not an object, skip it
                continue

            # Skip if no mask
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

            # Create segment mask in full image space
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

            # Validate mask_tensor shape
            if mask_tensor.ndim != 2:
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

            # Union this segment into the full mask
            full_mask = torch.max(full_mask, segment_mask)
            has_valid_mask = True

        # If no valid masks found, return empty queries
        if not has_valid_mask:
            return ("[]", "[]")

        # Find all non-zero pixels (use threshold for floating point masks)
        mask_pixels = full_mask > 0.5

        # Check if mask is empty
        if not mask_pixels.any():
            return ("[]", "[]")

        # Get coordinates of all mask pixels
        y_coords, x_coords = torch.where(mask_pixels)

        # Calculate bounding box (min/max coordinates)
        x_min = float(x_coords.min().item())
        y_min = float(y_coords.min().item())
        x_max = float(x_coords.max().item())
        y_max = float(y_coords.max().item())

        # Calculate centroid (center of mass)
        # Weight all mask pixels equally (binary mask)
        centroid_x = float(x_coords.float().mean().item())
        centroid_y = float(y_coords.float().mean().item())

        # Create box query (XYXY format)
        box_query = [{"x1": x_min, "y1": y_min, "x2": x_max, "y2": y_max}]

        # Create point query (centroid)
        point_query = [{"x": centroid_x, "y": centroid_y}]

        # Convert to JSON strings
        box_query_str = json.dumps(box_query)
        point_query_str = json.dumps(point_query)

        return (box_query_str, point_query_str)
