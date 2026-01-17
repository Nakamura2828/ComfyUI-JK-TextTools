"""
JSON Pretty Printer Node for ComfyUI

Formats JSON strings with proper indentation for readability.
Validates JSON and provides error messages for invalid input.
"""

import json


class JSONPrettyPrinter:
    """
    Format JSON strings with proper indentation.
    
    Takes raw JSON string and outputs formatted, readable JSON.
    Validates input and provides helpful error messages.
    
    Example Input:
    [{"name":"Alice","age":30},{"name":"Bob","age":25}]
    
    Example Output:
    [
      {
        "name": "Alice",
        "age": 30
      },
      {
        "name": "Bob",
        "age": 25
      }
    ]
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {
                    "default": "{}",
                    "multiline": True  # Allow pasting large JSON
                }),
                "indent": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 8,
                    "step": 1
                }),
            },
            "optional": {
                "sort_keys": ("BOOLEAN", {
                    "default": False  # Alphabetically sort object keys
                }),
                "compact": ("BOOLEAN", {
                    "default": False  # Minimize whitespace (for output, not display)
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("formatted_json", "is_valid", "error_message")
    FUNCTION = "format_json"
    CATEGORY = "JK-TextTools/json"
    
    def format_json(self, json_string, indent=2, sort_keys=False, compact=False):
        """
        Format JSON string with proper indentation.
        
        Args:
            json_string: Raw JSON string to format
            indent: Number of spaces for indentation (0 = compact)
            sort_keys: Whether to sort object keys alphabetically
            compact: If True, minimize whitespace (overrides indent)
            
        Returns:
            tuple: (formatted_json, is_valid, error_message)
        """
        try:
            # Parse JSON to validate
            parsed = json.loads(json_string)
            
            # Format with specified options
            if compact:
                # Compact format (no extra whitespace)
                formatted = json.dumps(parsed, separators=(',', ':'), sort_keys=sort_keys)
            else:
                # Pretty format with indentation
                formatted = json.dumps(parsed, indent=indent, sort_keys=sort_keys)
            
            return (formatted, True, "")
            
        except json.JSONDecodeError as e:
            # Provide helpful error message
            error_msg = f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}"
            
            # Return original string, invalid flag, and error
            return (json_string, False, error_msg)
            
        except Exception as e:
            # Catch any other errors
            error_msg = f"Error processing JSON: {str(e)}"
            return (json_string, False, error_msg)