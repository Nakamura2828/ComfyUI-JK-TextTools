"""
List Index Selector Node for ComfyUI

Extracts an item from a list by index.
Companion to String Splitter for list-based workflows.
"""


class ListIndexSelector:
    """
    Extract a single item from a list by index.
    
    Works with any list output, including from String Splitter.
    
    Example: ["10", "25", "42"] with index 1 → "25"
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("LIST", {
                    "forceInput": True  # Must be connected, no default
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
            },
            "optional": {
                "zero_indexed": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    # Can output either STRING or any type (using *)
    RETURN_TYPES = ("*", "INT")  
    RETURN_NAMES = ("selected_item", "list_length")
    FUNCTION = "select_from_list"
    CATEGORY = "JK-TextTools/list"
    
    def select_from_list(self, list_input, index, zero_indexed=True):
        """
        Extract item at specified index from a list.
        
        Args:
            list_input: A Python list
            index: Which item to extract
            zero_indexed: If True, 0 is first item. If False, 1 is first item.
            
        Returns:
            tuple: (selected_item, list_length)
        """
        # Adjust index if 1-indexed
        actual_index = index if zero_indexed else index - 1
        
        # Validate index
        if actual_index < 0 or actual_index >= len(list_input):
            # Return None and length if out of range
            return (None, len(list_input))
        
        selected = list_input[actual_index]
        length = len(list_input)
        
        return (selected, length)
