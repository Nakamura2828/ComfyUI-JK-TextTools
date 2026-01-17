"""
Detection Query Node for ComfyUI

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
    Returns filtered results as both JSON and list for different workflow needs.
    
    Examples:
        class_filter="CLASS1_LABEL" → exact match
        class_filter="CLASS1_*" → all CLASS1 subclasses
        class_filter="*" → all detections
        class_filter="*_LABEL" → all classes ending with _LABEL
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
                    "default": "*",  # Default to all classes
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
                    "default": "",  # Empty = don't extract
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", any_typ, any_typ, "BOOLEAN", "STRING")
    RETURN_NAMES = ("filtered_json", "match_count", "detection_list", "categorization_value", "is_valid", "error_message")
    OUTPUT_IS_LIST = (False, False, True, False, False, False)  # detection_list is a list
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
            categorization_field: Optional field name to extract (e.g., "is_dog", "categorization")
            
        Returns:
            tuple: (filtered_json, match_count, detection_list, categorization_value, is_valid, error_message)
        """
        try:
            # Parse JSON
            data = json.loads(json_string)
            
            # Handle common detection formats
            # Format 1: [{"detect_result": [...], ...}]
            # Format 2: {"detect_result": [...], ...}
            # Format 3: [detection, detection, ...]
            
            detections = []
            root_object = None  # Store root object for categorization extraction
            
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    if "detect_result" in data[0]:
                        # Format 1: Wrapped in object with detect_result key
                        root_object = data[0]
                        detections = data[0]["detect_result"]
                    else:
                        # Format 3: Direct list of detections
                        detections = data
            elif isinstance(data, dict):
                if "detect_result" in data:
                    # Format 2: Object with detect_result key
                    root_object = data
                    detections = data["detect_result"]
            
            # Extract categorization field if requested
            categorization_value = None
            if categorization_field and root_object:
                categorization_value = root_object.get(categorization_field, None)
            
            # Filter detections
            filtered = []
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
            
            # Apply max_results limit
            if max_results > 0:
                filtered = filtered[:max_results]
            
            # Prepare outputs
            count = len(filtered)
            
            # JSON output (wrapped in same format as input)
            if isinstance(data, list) and len(data) > 0 and "detect_result" in data[0]:
                # Maintain original wrapper format
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
                # Simple list format
                filtered_json = json.dumps(filtered, indent=2)
            
            # List output (for iteration)
            detection_list = filtered  # OUTPUT_IS_LIST will handle this
            
            return (filtered_json, count, detection_list, categorization_value, True, "")
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON Error at line {e.lineno}, column {e.colno}: {e.msg}"
            return ("[]", 0, [], None, False, error_msg)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return ("[]", 0, [], None, False, error_msg)