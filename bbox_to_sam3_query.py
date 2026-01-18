"""
BBox to SAM3 Query Node for ComfyUI

Converts BBOX to TBG SAM3 Selector query formats.
Generates both box and point queries for SAM3 refinement.
"""

import json


class BBoxToSAM3Query:
    """
    Convert BBOX to SAM3 Selector query formats.

    Takes a BBOX in XYWH format and generates:
    - Box query (XYXY format) for TBG SAM3 Selector
    - Point query (center point) for TBG SAM3 Selector

    BBOX format: [[x, y, width, height]]
    Box query: [{"x1": float, "y1": float, "x2": float, "y2": float}]
    Point query: [{"x": float, "y": float}]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("box_query", "point_query")
    FUNCTION = "bbox_to_sam3_query"
    CATEGORY = "JK-TextTools/bbox"

    def bbox_to_sam3_query(self, bbox):
        """
        Convert BBOX to SAM3 query formats.

        Args:
            bbox: [[x, y, w, h]] format (XYWH)

        Returns:
            tuple: (box_query_string, point_query_string)
        """
        # Validate and unwrap bbox
        if not isinstance(bbox, list) or len(bbox) == 0:
            # Empty bbox - return empty queries
            return ("[]", "[]")

        # Handle both [[x,y,w,h]] and [x,y,w,h] formats
        if isinstance(bbox[0], list):
            box = bbox[0]
        else:
            box = bbox

        if len(box) != 4:
            # Invalid bbox - return empty queries
            return ("[]", "[]")

        # Extract coordinates (XYWH format)
        x, y, w, h = box
        x, y, w, h = float(x), float(y), float(w), float(h)

        # Convert XYWH to XYXY for box query
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        # Calculate center point for point query
        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)

        # Create box query (XYXY format)
        box_query = [{"x1": x1, "y1": y1, "x2": x2, "y2": y2}]

        # Create point query (center point)
        point_query = [{"x": center_x, "y": center_y}]

        # Convert to JSON strings
        box_query_str = json.dumps(box_query)
        point_query_str = json.dumps(point_query)

        return (box_query_str, point_query_str)
