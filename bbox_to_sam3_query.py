"""
BBox to SAM3 Query Node for ComfyUI

Converts BBOX to both SAM3 and TBG SAM3 Selector query formats.
Generates both box and point queries for SAM3 refinement.
"""

import json


class BBoxToSAM3Query:
    """
    Convert BBOX to SAM3 query formats.

    Takes a BBOX in XYWH format and generates:
    - SAM3 format: Normalized coordinates with label arrays
    - TBG SAM3 Selector format: Absolute coordinates

    BBOX format: [[x, y, width, height]]

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
                "bbox": ("BBOX", {}),
            },
            "optional": {
                "width": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "height": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "prompt_type": (["positive", "negative"], {"default": "positive"}),
            },
        }

    RETURN_TYPES = ("SAM3_BOXES_PROMPT", "SAM3_POINTS_PROMPT", "STRING", "STRING")
    RETURN_NAMES = ("box_sam3", "point_sam3", "box_tbg_sam3", "point_tbg_sam3")
    FUNCTION = "bbox_to_sam3_query"
    CATEGORY = "JK-TextTools/bbox"

    def bbox_to_sam3_query(self, bbox, width=0, height=0, prompt_type="positive"):
        """
        Convert BBOX to SAM3 query formats.

        Args:
            bbox: [[x, y, w, h]] format (XYWH)
            width: Image width (required for SAM3 normalized coordinates)
            height: Image height (required for SAM3 normalized coordinates)
            prompt_type: "positive" or "negative" (for SAM3 labels)

        Returns:
            tuple: (box_prompt_dict, point_prompt_dict, box_query_tbg_string, point_query_tbg_string)
        """
        # Unwrap optional parameters if they come as lists
        if isinstance(width, list):
            width = width[0] if len(width) > 0 else 0
        if isinstance(height, list):
            height = height[0] if len(height) > 0 else 0
        if isinstance(prompt_type, list):
            prompt_type = prompt_type[0]

        # Validate and unwrap bbox
        if not isinstance(bbox, list) or len(bbox) == 0:
            # Empty bbox - return empty queries
            empty_box_prompt = {"boxes": [], "labels": []}
            empty_point_prompt = {"points": [], "labels": []}
            return (empty_box_prompt, empty_point_prompt, "[]", "[]")

        # Handle both [[x,y,w,h]] and [x,y,w,h] formats
        if isinstance(bbox[0], list):
            box = bbox[0]
        else:
            box = bbox

        if len(box) != 4:
            # Invalid bbox - return empty queries
            empty_box_prompt = {"boxes": [], "labels": []}
            empty_point_prompt = {"points": [], "labels": []}
            return (empty_box_prompt, empty_point_prompt, "[]", "[]")

        # Extract coordinates (XYWH format)
        x, y, w, h = box
        x, y, w, h = float(x), float(y), float(w), float(h)

        # Convert XYWH to XYXY for TBG box query
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        # Calculate center point for point query
        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)

        # === SAM3 Format Outputs (normalized coordinates) ===
        # Check if width/height are provided for normalization
        if width > 0 and height > 0:
            # Normalize coordinates to 0-1 range
            x_norm = x / width
            y_norm = y / height
            w_norm = w / width
            h_norm = h / height

            # Normalize center point
            center_x_norm = center_x / width
            center_y_norm = center_y / height

            # Determine labels based on prompt type
            if prompt_type == "positive":
                box_label = True
                point_label = 1
            else:  # negative
                box_label = False
                point_label = 0

            # Create SAM3 format prompts (XYWH normalized)
            box_prompt = {
                "boxes": [[x_norm, y_norm, w_norm, h_norm]],
                "labels": [box_label]
            }

            point_prompt = {
                "points": [[center_x_norm, center_y_norm]],
                "labels": [point_label]
            }
        else:
            # Width/height not provided - return empty SAM3 outputs
            box_prompt = {"boxes": [], "labels": []}
            point_prompt = {"points": [], "labels": []}

        # === TBG SAM3 Selector Format Outputs (absolute coordinates) ===
        # Create TBG box query (XYXY format)
        box_query_tbg = [{"x1": x1, "y1": y1, "x2": x2, "y2": y2}]

        # Create TBG point query (center point)
        point_query_tbg = [{"x": center_x, "y": center_y}]

        # Convert TBG outputs to JSON strings
        box_query_tbg_str = json.dumps(box_query_tbg)
        point_query_tbg_str = json.dumps(point_query_tbg)

        return (box_prompt, point_prompt, box_query_tbg_str, point_query_tbg_str)
