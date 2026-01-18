"""
Mask to BBox Node for ComfyUI

Converts binary masks to bounding boxes in XYWH format.
Useful for nodes that output rectangular masks instead of proper BBOX type.
"""

import torch


class MaskToBBox:
    """
    Convert mask to bounding box.

    Takes a binary mask and generates:
    - BBox in XYWH format (for chaining to other bbox nodes)
    - Individual x, y, w, h coordinates (integer pixel values)

    Mask threshold: Pixels > 0.5 are considered part of the mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {}),
            },
        }

    RETURN_TYPES = ("BBOX", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("bbox", "x", "y", "w", "h")
    FUNCTION = "mask_to_bbox"
    CATEGORY = "JK-TextTools/mask"

    def mask_to_bbox(self, mask):
        """
        Convert mask to bounding box.

        Args:
            mask: Binary mask tensor (height, width)

        Returns:
            tuple: (bbox, x, y, w, h)
        """
        # Validate mask
        if not isinstance(mask, torch.Tensor):
            # Invalid mask - return empty bbox
            return ([[0, 0, 0, 0]], 0, 0, 0, 0)

        # Handle batch dimension if present
        if mask.ndim == 3:
            # Take first mask if batched
            mask = mask[0]

        if mask.ndim != 2:
            # Invalid dimensions - return empty bbox
            return ([[0, 0, 0, 0]], 0, 0, 0, 0)

        # Find all non-zero pixels (threshold at 0.5 for float masks)
        mask_pixels = mask > 0.5

        # Check if mask is empty
        if not mask_pixels.any():
            # Empty mask - return zero bbox
            return ([[0, 0, 0, 0]], 0, 0, 0, 0)

        # Get coordinates of all mask pixels
        y_coords, x_coords = torch.where(mask_pixels)

        # Calculate bounding box (min/max coordinates)
        x_min = int(x_coords.min().item())
        y_min = int(y_coords.min().item())
        x_max = int(x_coords.max().item())
        y_max = int(y_coords.max().item())

        # Calculate width and height (inclusive of max pixel)
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        # Create BBOX in XYWH format
        bbox = [[x_min, y_min, width, height]]

        return (bbox, x_min, y_min, width, height)
