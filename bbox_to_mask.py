"""
BBox to Mask Node for ComfyUI

Converts a single bounding box to a binary mask.
Simple 1:1 conversion - for multiple bboxes, use BBoxesToMask.
"""

import torch


class BBoxToMask:
    """
    Convert a single bounding box to a binary mask.

    Takes one bbox and image dimensions, outputs one mask.
    When connected to list outputs, ComfyUI iterates automatically.

    For combined/union masks from multiple bboxes, use BBoxesToMask instead.

    Bbox format: [[x, y, width, height]]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {}),
                "width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
            },
            "optional": {
                "invert": ("BOOLEAN", {
                    "default": False  # If True, bbox is black, rest is white
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "bbox_to_mask"
    CATEGORY = "JK-TextTools/bbox"
    
    def bbox_to_mask(self, bbox, width, height, invert=False):
        """
        Convert a single bbox to a mask.

        Args:
            bbox: [[x, y, w, h]] format
            width: Image width
            height: Image height
            invert: If True, bbox area is 0, rest is 1

        Returns:
            tuple: (mask,)
        """
        # Validate input
        if not isinstance(bbox, list) or len(bbox) == 0:
            # Empty bbox - return empty mask
            empty_mask = torch.zeros((height, width), dtype=torch.float32)
            return (empty_mask,)

        # Unwrap bbox - handle both [[x,y,w,h]] and [x,y,w,h] formats
        if isinstance(bbox[0], list):
            # It's [[x,y,w,h]] format, unwrap it
            box = bbox[0]
        else:
            # It's [x,y,w,h] format
            box = bbox

        if len(box) != 4:
            # Invalid bbox - return empty mask
            empty_mask = torch.zeros((height, width), dtype=torch.float32)
            return (empty_mask,)

        # Extract coordinates
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Create mask
        mask = torch.zeros((height, width), dtype=torch.float32)

        # Clamp coordinates to image bounds
        x1 = max(0, min(x, width))
        y1 = max(0, min(y, height))
        x2 = max(0, min(x + w, width))
        y2 = max(0, min(y + h, height))

        # Fill bbox area
        if x2 > x1 and y2 > y1:
            if invert:
                # Entire mask is white except bbox
                mask.fill_(1.0)
                mask[y1:y2, x1:x2] = 0.0
            else:
                # Only bbox is white
                mask[y1:y2, x1:x2] = 1.0

        return (mask,)