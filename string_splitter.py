"""
String Splitter Node for ComfyUI

Splits a delimited string into a list with optional type casting.
"""

# wildcard trick is taken from pythongossss's & ImpactPack
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


class StringSplitter:
    """
    Split a delimited string into a list of strings, integers, or floats.
    
    Returns a list type that can be consumed by:
    - List iteration nodes
    - List index selector nodes
    - Other list-aware nodes
    
    Example: "10,25,42,100" → [10, 25, 42, 100] (if output_type=INT)
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
                "output_type": (["STRING", "INT", "FLOAT"],),  # Type to cast to
            },
            "optional": {
                "remove_empty": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = (any_typ, "INT")
    RETURN_NAMES = ("list", "item_count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "split_string"
    CATEGORY = "JK-TextTools/string"
    
    def split_string(self, text, delimiter, strip_whitespace=True, output_type="STRING", remove_empty=False):
        """
        Split string into a list with optional type casting.
        
        Args:
            text: The delimited string
            delimiter: What to split on
            strip_whitespace: Clean up each item
            output_type: Type to cast items to ("STRING", "INT", or "FLOAT")
            remove_empty: Remove empty strings from result
            
        Returns:
            tuple: (list_of_items, count)
            
        Raises:
            ValueError: If type casting fails for any item
        """
        # Handle empty string edge case
        if not text or (strip_whitespace and not text.strip()):
            return ([], 0)
        
        # Process common escape sequences
        # This allows typing \n in UI to get actual newlines
        escape_map = {
            '\\n': '\n',
            '\\t': '\t',
            '\\r': '\r',
            '\\\\': '\\',
        }
        
        for escape_seq, actual_char in escape_map.items():
            delimiter = delimiter.replace(escape_seq, actual_char)
                
        # Split the string
        items = text.split(delimiter)
        
        # Strip whitespace if requested
        if strip_whitespace:
            items = [item.strip() for item in items]
        
        # Remove empty strings if requested (before type casting)
        if remove_empty:
            items = [item for item in items if item]
        
        # Type casting
        try:
            if output_type == "INT":
                items = [int(item) for item in items]
            elif output_type == "FLOAT":
                items = [float(item) for item in items]
            # STRING is default, no casting needed
        except ValueError as e:
            # Provide helpful error message
            raise ValueError(
                f"Failed to convert item to {output_type}. "
                f"Check that all items in '{text}' can be converted to {output_type}. "
                f"Error: {e}"
            )
        
        count = len(items)
        
        return (items, count)