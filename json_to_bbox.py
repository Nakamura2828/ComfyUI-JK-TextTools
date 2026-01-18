"""
JSON to BBox Node for ComfyUI

Converts JSON string representation of bboxes to proper BBOX format.
Designed for nodes that output bboxes as JSON strings (like SAM3).
"""

import json


class JSONToBBox:
    """
    Convert JSON string of bboxes to BBOX format.
    
    Handles JSON arrays like:
    [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
    
    Can convert between coordinate formats:
    - XYXY (x1, y1, x2, y2) - two corners
    - XYWH (x, y, width, height) - corner + dimensions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {
                    "default": "[]",
                    "multiline": True
                }),
                "input_format": (["XYXY", "XYWH"],),  # Format in the JSON
                "output_format": (["XYXY", "XYWH"],),  # Format to output
            },
        }
    
    RETURN_TYPES = ("*", "INT")
    RETURN_NAMES = ("bboxes", "bbox_count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "json_to_bbox"
    CATEGORY = "JK-TextTools/bbox"
    
    def json_to_bbox(self, json_string, input_format="XYXY", output_format="XYWH"):
        """
        Convert JSON string to bbox list.
        
        Args:
            json_string: JSON array of bboxes
            input_format: Format in JSON ("XYXY" or "XYWH")
            output_format: Format to output ("XYXY" or "XYWH")
            
        Returns:
            tuple: (bbox_list, count)
        """
        try:
            # Parse JSON
            data = json.loads(json_string)
            
            if not isinstance(data, list):
                # Not a list - return empty
                return ([], 0)
            
            bboxes = []
            
            for bbox in data:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                
                # Convert to floats/ints
                bbox = [float(x) for x in bbox]
                
                # Convert format if needed
                if input_format == "XYXY" and output_format == "XYWH":
                    # Convert (x1, y1, x2, y2) to (x, y, w, h)
                    x1, y1, x2, y2 = bbox
                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1
                    bbox = [x, y, w, h]
                    
                elif input_format == "XYWH" and output_format == "XYXY":
                    # Convert (x, y, w, h) to (x1, y1, x2, y2)
                    x, y, w, h = bbox
                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h
                    bbox = [x1, y1, x2, y2]
                
                # Wrap in list for BBOX format: [[x,y,w,h]]
                bboxes.append([bbox])
            
            return (bboxes, len(bboxes))
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return ([], 0)
        except Exception as e:
            print(f"Error converting JSON to bbox: {e}")
            return ([], 0)
