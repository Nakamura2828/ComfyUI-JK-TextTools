"""
SEGs to SAM3 Query Node for ComfyUI

Converts SEGS segmentation to both SAM3 and TBG SAM3 Selector query formats.
Generates box query (bounding box) and point query (mask centroid).
"""

import json
import torch
import numpy as np


class SEGsToSAM3Query:
    """
    Convert SEGS segmentation to SAM3 query formats.

    Takes SEGS and generates:
    - SAM3 format: Normalized coordinates with label arrays
    - TBG SAM3 Selector format: Absolute coordinates

    SEGS format: ((height, width), [SEG(...), SEG(...), ...])

    SAM3 format outputs:
    - box_prompt: {"boxes": [[x_norm, y_norm, w_norm, h_norm]], "labels": [True/False]}
    - point_prompt: {"points": [[x_norm, y_norm]], "labels": [1/0]}

    TBG format outputs:
    - box_query_tbg: [{"x1": float, "y1": float, "x2": float, "y2": float}]
    - point_query_tbg: [{"x": float, "y": float}]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS", {}),
            },
            "optional": {
                "prompt_type": (["positive", "negative"], {"default": "positive"}),
            },
        }

    RETURN_TYPES = ("SAM3_BOXES_PROMPT", "SAM3_POINTS_PROMPT", "STRING", "STRING")
    RETURN_NAMES = ("box_sam3", "point_sam3", "box_tbg_sam3", "point_tbg_sam3")
    FUNCTION = "segs_to_sam3_query"
    CATEGORY = "JK-TextTools/segs"

    def segs_to_sam3_query(self, segs, prompt_type="positive"):
        """
        Convert SEGS to SAM3 query formats.

        Args:
            segs: SEGS tuple ((height, width), [SEG(...), ...])
            prompt_type: "positive" or "negative" (for SAM3 labels)

        Returns:
            tuple: (box_prompt_dict, point_prompt_dict, box_query_tbg_string, point_query_tbg_string)
        """
        # Unwrap prompt_type if it comes as a list (from optional param)
        if isinstance(prompt_type, list):
            prompt_type = prompt_type[0]

        # Validate SEGS format
        if not isinstance(segs, tuple) or len(segs) != 2:
            # Invalid SEGS format - return empty queries
            empty_box_prompt = {"boxes": [], "labels": []}
            empty_point_prompt = {"points": [], "labels": []}
            return (empty_box_prompt, empty_point_prompt, "[]", "[]")

        dims, seg_list = segs

        # Extract dimensions
        if isinstance(dims, tuple) and len(dims) == 2:
            height, width = dims
        else:
            # Invalid dimensions - return empty
            empty_box_prompt = {"boxes": [], "labels": []}
            empty_point_prompt = {"points": [], "labels": []}
            return (empty_box_prompt, empty_point_prompt, "[]", "[]")

        # Validate seg_list
        if not isinstance(seg_list, list) or len(seg_list) == 0:
            # Empty seg list - return empty queries
            empty_box_prompt = {"boxes": [], "labels": []}
            empty_point_prompt = {"points": [], "labels": []}
            return (empty_box_prompt, empty_point_prompt, "[]", "[]")

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
            empty_box_prompt = {"boxes": [], "labels": []}
            empty_point_prompt = {"points": [], "labels": []}
            return (empty_box_prompt, empty_point_prompt, "[]", "[]")

        # Find all non-zero pixels (use threshold for floating point masks)
        mask_pixels = full_mask > 0.5

        # Check if mask is empty
        if not mask_pixels.any():
            empty_box_prompt = {"boxes": [], "labels": []}
            empty_point_prompt = {"points": [], "labels": []}
            return (empty_box_prompt, empty_point_prompt, "[]", "[]")

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

        # === SAM3 Format Outputs (normalized coordinates) ===
        # Normalize coordinates to 0-1 range
        x_min_norm = x_min / width
        y_min_norm = y_min / height
        x_max_norm = x_max / width
        y_max_norm = y_max / height

        # Calculate width and height (normalized)
        w_norm = x_max_norm - x_min_norm
        h_norm = y_max_norm - y_min_norm

        # Normalize centroid
        centroid_x_norm = centroid_x / width
        centroid_y_norm = centroid_y / height

        # Determine labels based on prompt type
        if prompt_type == "positive":
            box_label = True
            point_label = 1
        else:  # negative
            box_label = False
            point_label = 0

        # Create SAM3 format prompts (XYWH normalized)
        box_prompt = {
            "boxes": [[x_min_norm, y_min_norm, w_norm, h_norm]],
            "labels": [box_label]
        }

        point_prompt = {
            "points": [[centroid_x_norm, centroid_y_norm]],
            "labels": [point_label]
        }

        # === TBG SAM3 Selector Format Outputs (absolute coordinates) ===
        # Create TBG box query (XYXY format)
        box_query_tbg = [{"x1": x_min, "y1": y_min, "x2": x_max, "y2": y_max}]

        # Create TBG point query (centroid)
        point_query_tbg = [{"x": centroid_x, "y": centroid_y}]

        # Convert TBG outputs to JSON strings
        box_query_tbg_str = json.dumps(box_query_tbg)
        point_query_tbg_str = json.dumps(point_query_tbg)

        return (box_prompt, point_prompt, box_query_tbg_str, point_query_tbg_str)
