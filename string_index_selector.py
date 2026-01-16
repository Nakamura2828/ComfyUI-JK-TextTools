"""
String Index Selector Node for ComfyUI

Extract a single element from a delimited string by index.
Perfect for loop iterations where you need the Nth item.
"""


class StringIndexSelector:
    """
    Extract a single element from a delimited string by index.
    
    Use case: In a loop, connect loop index to get corresponding value.
    Example: "10,25,42,100" with index 2 → "42"
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "item1,item2,item3",
                    "multiline": False
                }),
                "delimiter": ("STRING", {
                    "default": ",",
                    "multiline": False
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "strip_whitespace": ("BOOLEAN", {
                    "default": True
                }),
            },
            "optional": {
                "zero_indexed": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("selected_item", "item_count")
    FUNCTION = "select_by_index"
    CATEGORY = "JK-TextTools/string"
    
    def select_by_index(self, text, delimiter, index, strip_whitespace=True, zero_indexed=True):
        """
        Extract item at specified index from delimited string.
        
        Args:
            text: The delimited string
            delimiter: Character(s) to split on
            index: Which item to extract
            strip_whitespace: Remove leading/trailing spaces from each item
            zero_indexed: If True, 0 is first item. If False, 1 is first item.
            
        Returns:
            tuple: (selected_item, total_count)
        """
        # Split the string
        items = text.split(delimiter)
        
        # Strip whitespace if requested
        if strip_whitespace:
            items = [item.strip() for item in items]
        
        # Adjust index if 1-indexed
        actual_index = index if zero_indexed else index - 1
        
        # Validate index and handle out of range gracefully
        if actual_index < 0 or actual_index >= len(items):
            # Return empty string instead of crashing
            return ("", len(items))
        
        selected = items[actual_index]
        count = len(items)
        
        return (selected, count)
