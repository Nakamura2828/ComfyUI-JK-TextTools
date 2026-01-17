"""
Detection to BBox Node for ComfyUI

Extracts bounding box from detection JSON and converts to BBOX format.
Compatible with KJNodes BBox visualizer and other bbox-aware nodes.
"""

import json


class DetectionToBBox:
    """
    Extract bounding box from detection object.
    
    Takes a single detection dict (from Detection Query's detection_list)
    and extracts the bbox in a format compatible with bbox visualization nodes.
    
    Input detection format:
    {
        "class": "CLASS_NAME",
        "score": 0.95,
        "box": [x, y, width, height]  # or "bbox"
    }
    
    Output: [x, y, width, height] as separate values or list
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detection": ("STRING", {
                    "default": "{}",
                    "multiline": True
                }),
            },
            "optional": {
                "bbox_key": (["box", "bbox"],),  # Which key to look for
            }
        }
    
    RETURN_TYPES = ("BBOX", "INT", "INT", "INT", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("bbox", "x", "y", "width", "height", "class_name", "score")
    FUNCTION = "extract_bbox"
    CATEGORY = "JK-TextTools/json"
    
    def extract_bbox(self, detection, bbox_key="box"):
        """
        Extract bounding box from detection object.
        
        Args:
            detection: JSON string of detection object
            bbox_key: Which key contains the bbox ("box" or "bbox")
            
        Returns:
            tuple: (bbox_list, x, y, width, height, class_name, score)
        """
        try:
            # Parse detection if it's a string
            if isinstance(detection, str):
                det = json.loads(detection)
            else:
                det = detection
            
            # Extract bbox
            if bbox_key in det:
                bbox = det[bbox_key]
            elif "box" in det:
                bbox = det["box"]
            elif "bbox" in det:
                bbox = det["bbox"]
            else:
                # No bbox found - return zeros but still extract class/score
                class_name = det.get("class", "")
                score = det.get("score", 0.0)
                return ([[0, 0, 0, 0]], 0, 0, 0, 0, class_name, float(score))
            
            # Ensure bbox has 4 values
            if len(bbox) != 4:
                # Invalid bbox - return zeros but still extract class/score
                class_name = det.get("class", "")
                score = det.get("score", 0.0)
                return ([[0, 0, 0, 0]], 0, 0, 0, 0, class_name, float(score))
            
            x, y, w, h = bbox
            
            # Extract class and score if available
            class_name = det.get("class", "")
            score = det.get("score", 0.0)
            
            # BBOX format for KJNodes: list of lists [[x, y, w, h]]
            bbox_list = [[int(x), int(y), int(w), int(h)]]
            
            # Return bbox as nested list, plus individual values
            return (bbox_list, int(x), int(y), int(w), int(h), class_name, float(score))
            
        except Exception as e:
            # Return zeros on error
            print(f"Error extracting bbox: {e}")
            return ([[0, 0, 0, 0]], 0, 0, 0, 0, "", 0.0)