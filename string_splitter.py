"""
String Splitter Node for ComfyUI

Splits a delimited string into a list that can be consumed by other nodes.
"""

# wildcard trick is taken from pythongossss's & ImpactPack
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class StringSplitter:
    """
    Split a delimited string into a list of strings.
    
    Returns a list type that can be consumed by:
    - List iteration nodes
    - List index selector nodes
    - Other list-aware nodes
    
    Example: "10,25,42,100" → ["10", "25", "42", "100"]
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
                "strip_whitespace": ("BOOLEAN", {
                    "default": True
                }),
            },
            "optional": {
                "remove_empty": ("BOOLEAN", {
                    "default": False  # Remove empty strings from result
                }),
            }
        }
    
    # Try LIST as the output type
    RETURN_TYPES = (any_typ, "INT")
    RETURN_NAMES = ("list", "item_count")
    OUTPUT_IS_LIST = (True,False)
    FUNCTION = "split_string"
    CATEGORY = "JK-TextTools/string"
    
    def split_string(self, text, delimiter, strip_whitespace=True, remove_empty=False):
        """
        Split string into a list.
        
        Args:
            text: The delimited string
            delimiter: What to split on
            strip_whitespace: Clean up each item
            remove_empty: Remove empty strings from result
            
        Returns:
            tuple: (list_of_strings, count)
        """
        # Split the string
        items = text.split(delimiter)
        
        # Strip whitespace if requested
        if strip_whitespace:
            items = [item.strip() for item in items]
        
        # Remove empty strings if requested
        if remove_empty:
            items = [item for item in items if item]
        
        count = len(items)
        
        return (items, count)
