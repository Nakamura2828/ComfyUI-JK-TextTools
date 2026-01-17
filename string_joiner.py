"""
String Joiner Node for ComfyUI

Joins a list of items into a single delimited string.
Companion to String Splitter - reverses the split operation.
"""

# wildcard trick is taken from pythongossss's & ImpactPack
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


class StringJoiner:
    """
    Join a list of items into a single delimited string.
    
    Works with any list output, including from String Splitter.
    Converts all items to strings before joining.
    
    Example: [10, 25, 42, 100] with delimiter "," → "10,25,42,100"
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": (any_typ, {"forceInput": True}),  # Accept any list
                "delimiter": ("STRING", {
                    "default": ",",
                    "multiline": False
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("joined_string", "item_count")
    INPUT_IS_LIST = True  # Receive lists
    FUNCTION = "join_list"
    CATEGORY = "JK-TextTools/string"
    
    def join_list(self, list_input, delimiter):
        """
        Join list items into a delimited string.
        
        Args:
            list_input: A Python list from OUTPUT_IS_LIST
            delimiter: String to insert between items
            
        Returns:
            tuple: (joined_string, item_count)
        """
        # When INPUT_IS_LIST = True, all inputs come as lists
        # Unwrap scalar inputs
        if isinstance(delimiter, list):
            delimiter = delimiter[0] if delimiter else ","
        
        # Convert all items to strings
        str_items = [str(item) for item in list_input]
        
        # Join with delimiter
        result = delimiter.join(str_items)
        count = len(list_input)
        
        return (result, count)