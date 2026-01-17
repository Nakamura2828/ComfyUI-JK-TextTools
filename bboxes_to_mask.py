"""
BBoxes to Mask Node for ComfyUI

Converts a list of bounding boxes to masks.
Takes the full bbox list (not OUTPUT_IS_LIST) and creates combined mask.
"""

import torch


class BBoxesToMask:
    """
    Convert multiple bounding boxes to masks.
    
    Takes a list of bboxes and creates:
    - Combined mask (union of all bboxes)
    - Individual masks for each bbox (as list)
    
    This node expects the bbox_list WITHOUT OUTPUT_IS_LIST unwrapping.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("*", {"forceInput": True}),  # Accept list of bboxes
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
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK", "INT")
    RETURN_NAMES = ("combined_mask", "individual_masks", "bbox_count")
    OUTPUT_IS_LIST = (False, True, False)  # individual_masks is a list
    INPUT_IS_LIST = True  # Accept lists for all inputs
    FUNCTION = "bboxes_to_mask"
    CATEGORY = "JK-TextTools/bbox"
    
    def bboxes_to_mask(self, bboxes, width, height, invert=False):
        """
        Convert list of bboxes to masks.
        
        Args:
            bboxes: List of bboxes (comes as list due to INPUT_IS_LIST)
            width: Image width (comes as list, unwrap it)
            height: Image height (comes as list, unwrap it)
            invert: If True, bbox areas are 0, rest is 1 (comes as list, unwrap it)
            
        Returns:
            tuple: (combined_mask, list_of_individual_masks, count)
        """
        # Unwrap scalar inputs that come as lists due to INPUT_IS_LIST
        if isinstance(width, list):
            width = width[0] if width else 512
        if isinstance(height, list):
            height = height[0] if height else 512
        if isinstance(invert, list):
            invert = invert[0] if invert else False
        
        # bboxes is already a list from OUTPUT_IS_LIST upstream
        if not isinstance(bboxes, list) or len(bboxes) == 0:
            # Empty input - return empty masks
            empty_mask = torch.zeros((height, width), dtype=torch.float32)
            return (empty_mask, [empty_mask], 0)
        
        # Normalize bbox format - handle both [[x,y,w,h]...] and [[[x,y,w,h]]...]
        normalized_bboxes = []
        for bbox in bboxes:
            if isinstance(bbox, list):
                # Check if it's wrapped [[x,y,w,h]] or unwrapped [x,y,w,h]
                if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                    # Unwrapped [x,y,w,h]
                    normalized_bboxes.append(bbox)
                elif len(bbox) > 0 and isinstance(bbox[0], list) and len(bbox[0]) == 4:
                    # Wrapped [[x,y,w,h]]
                    normalized_bboxes.append(bbox[0])
        
        if len(normalized_bboxes) == 0:
            # No valid bboxes
            empty_mask = torch.zeros((height, width), dtype=torch.float32)
            return (empty_mask, [empty_mask], 0)
        
        # Create combined mask
        combined_mask = torch.zeros((height, width), dtype=torch.float32)
        individual_masks = []
        
        for bbox in normalized_bboxes:
            if len(bbox) != 4:
                continue
            
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Clamp to image bounds
            x1 = max(0, min(x, width))
            y1 = max(0, min(y, height))
            x2 = max(0, min(x + w, width))
            y2 = max(0, min(y + h, height))
            
            # Create individual mask
            individual_mask = torch.zeros((height, width), dtype=torch.float32)
            
            if x2 > x1 and y2 > y1:
                # Fill individual mask
                individual_mask[y1:y2, x1:x2] = 1.0
                
                # Add to combined mask (union)
                combined_mask[y1:y2, x1:x2] = 1.0
            
            individual_masks.append(individual_mask)
        
        # Apply invert if needed
        if invert:
            combined_mask = 1.0 - combined_mask
            individual_masks = [1.0 - mask for mask in individual_masks]
        
        count = len(individual_masks)
        
        return (combined_mask, individual_masks, count)
