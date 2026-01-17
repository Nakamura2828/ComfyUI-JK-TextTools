"""
Detection Query Node for ComfyUI - UPDATED with bbox_list output

Query detection results with class filtering, score thresholds, and wildcards.
Designed for object detection/classification workflow outputs.
"""

import json
import fnmatch


# Wildcard trick from ImpactPack/pythongossss
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


class DetectionQuery:
    """
    Query detection results with filtering and wildcards.
    
    Filters detection data by class name (with wildcards) and confidence score.
    Returns filtered results as JSON, list, and bbox list for visualization.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {
                    "default": "[]",
                    "multiline": True
                }),
                "class_filter": ("STRING", {
                    "default": "*",
                    "multiline": False
                }),
            },
            "optional": {
                "min_score": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "max_results": ("INT", {
                    "default": 0,  # 0 = unlimited
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "categorization_field": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", any_typ, "BBOX", any_typ, "BOOLEAN", "STRING")
    RETURN_NAMES = ("filtered_json", "match_count", "detection_list", "bbox_list", "categorization_value", "is_valid", "error_message")
    OUTPUT_IS_LIST = (False, False, True, True, False, False, False)  # detection_list AND bbox_list are lists
    FUNCTION = "query_detections"
    CATEGORY = "JK-TextTools/json"
    
    def query_detections(self, json_string, class_filter="*", min_score=0.0, max_results=0, categorization_field=""):
        """
        Query detection results with filtering.
        
        Args:
            json_string: JSON string containing detection results
            class_filter: Class name with wildcards (* and ?)
            min_score: Minimum confidence score (0.0-1.0)
            max_results: Maximum results to return (0 = unlimited)
            categorization_field: Optional field name to extract
            
        Returns:
            tuple: (filtered_json, match_count, detection_list, bbox_list, categorization_value, is_valid, error_message)
        """
        try:
            # Parse JSON
            data = json.loads(json_string)
            
            # Handle common detection formats
            detections = []
            root_object = None
            
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    if "detect_result" in data[0]:
                        root_object = data[0]
                        detections = data[0]["detect_result"]
                    else:
                        detections = data
            elif isinstance(data, dict):
                if "detect_result" in data:
                    root_object = data
                    detections = data["detect_result"]
            
            # Extract categorization field if requested
            categorization_value = None
            if categorization_field and root_object:
                categorization_value = root_object.get(categorization_field, None)
            
            # Filter detections
            filtered = []
            bbox_list = []  # Collect bboxes as we filter
            
            for detection in detections:
                # Check if detection has required fields
                if not isinstance(detection, dict):
                    continue
                if "class" not in detection or "score" not in detection:
                    continue
                
                class_name = detection["class"]
                score = detection["score"]
                
                # Apply class filter (with wildcards)
                if not fnmatch.fnmatch(class_name, class_filter):
                    continue
                
                # Apply score filter
                if score < min_score:
                    continue
                
                filtered.append(detection)
                
                # Extract bbox if present
                bbox = None
                if "box" in detection:
                    bbox = detection["box"]
                elif "bbox" in detection:
                    bbox = detection["bbox"]
                
                # Add to bbox_list in proper format
                # For OUTPUT_IS_LIST, each bbox needs to be wrapped: [[x,y,w,h]]
                if bbox and len(bbox) == 4:
                    # Wrap each bbox in a list for BBOX format
                    bbox_list.append([[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]])
                else:
                    # No valid bbox - add zeros (wrapped)
                    bbox_list.append([[0, 0, 0, 0]])
            
            # Apply max_results limit
            if max_results > 0:
                filtered = filtered[:max_results]
                bbox_list = bbox_list[:max_results]
            
            # Prepare outputs
            count = len(filtered)
            
            # JSON output (wrapped in same format as input)
            if isinstance(data, list) and len(data) > 0 and "detect_result" in data[0]:
                output_obj = {
                    "detect_result": filtered
                }
                # Preserve other fields from root object
                if root_object:
                    for key, value in root_object.items():
                        if key != "detect_result":
                            output_obj[key] = value
                filtered_json = json.dumps([output_obj], indent=2)
            else:
                filtered_json = json.dumps(filtered, indent=2)
            
            # List output (for iteration)
            detection_list = filtered  # OUTPUT_IS_LIST will handle this
            
            return (filtered_json, count, detection_list, bbox_list, categorization_value, True, "")
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON Error at line {e.lineno}, column {e.colno}: {e.msg}"
            return ("[]", 0, [], [], None, False, error_msg)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return ("[]", 0, [], [], None, False, error_msg)