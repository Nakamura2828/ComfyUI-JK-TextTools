"""
BBox to Mask Node for ComfyUI

Converts bounding boxes to binary masks.
Supports both individual masks and combined union mask.
"""

import torch


class BBoxToMask:
    """
    Convert bounding boxes to binary masks.
    
    Takes bbox(es) and image dimensions, outputs:
    - Individual masks for each bbox (as list)
    - Combined mask (union of all bboxes)
    
    Bbox format: [[x, y, width, height]] or list of bboxes
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
    
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("individual_masks", "combined_mask")
    OUTPUT_IS_LIST = (True, False)  # individual_masks is a list
    FUNCTION = "bbox_to_mask"
    CATEGORY = "JK-TextTools/bbox"
    
    def bbox_to_mask(self, bbox, width, height, invert=False):
        """
        Convert bbox(es) to mask(s).
        
        Args:
            bbox: [[x, y, w, h]] or [[[x1,y1,w1,h1]], [[x2,y2,w2,h2]], ...]
            width: Image width
            height: Image height
            invert: If True, bbox area is 0, rest is 1
            
        Returns:
            tuple: (list_of_masks, combined_mask)
        """
        # Handle both single bbox and list of bboxes
        # Single bbox: [[x, y, w, h]]
        # Multiple bboxes from OUTPUT_IS_LIST: already unwrapped to [[x,y,w,h]] per call
        
        if not isinstance(bbox, list) or len(bbox) == 0:
            # Empty bbox - return empty mask
            empty_mask = torch.zeros((height, width), dtype=torch.float32)
            return ([empty_mask], empty_mask)
        
        # Check if this is a single bbox [[x,y,w,h]] or already unwrapped [x,y,w,h]
        if isinstance(bbox[0], list):
            # It's [[x,y,w,h]] format
            bboxes = bbox
        else:
            # It's [x,y,w,h] format, wrap it
            bboxes = [bbox]
        
        individual_masks = []
        combined_mask = torch.zeros((height, width), dtype=torch.float32)
        
        for box in bboxes:
            if len(box) != 4:
                # Invalid bbox - skip
                continue
            
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Create individual mask
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
                
                # Add to combined mask (union)
                combined_mask[y1:y2, x1:x2] = 1.0
            
            individual_masks.append(mask)
        
        # Apply invert to combined mask if needed
        if invert:
            combined_mask = 1.0 - combined_mask
        
        # If no valid bboxes, return empty
        if len(individual_masks) == 0:
            empty_mask = torch.zeros((height, width), dtype=torch.float32)
            individual_masks = [empty_mask]
        
        return (individual_masks, combined_mask)