"""
List Index Selector Node for ComfyUI

Extracts an item from a list by index.
Companion to String Splitter for list-based workflows.
"""

# wildcard trick is taken from pythongossss's & ImpactPack
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


class ListIndexSelector:
    """
    Extract a single item from a list by index.
    
    Works with any list output, including from String Splitter.
    Returns the item in its original type (string, int, float, etc.)
    
    Example: [10, 25, 42] with index 1 → 25 (as integer)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": (any_typ, {"forceInput": True}),  # Accept any type, must be connected
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "zero_indexed": ("BOOLEAN", {
                    "default": True
                }),
            },
        }
    
    RETURN_TYPES = (any_typ, "INT")  # Use wildcard for selected item
    RETURN_NAMES = ("selected_item", "list_length")
    INPUT_IS_LIST = (True, False)
    FUNCTION = "select_from_list"
    CATEGORY = "JK-TextTools/list"
    
    def select_from_list(self, list_input, index, zero_indexed=True):
        """
        Extract item at specified index from a list.
        
        Args:
            list_input: A Python list from OUTPUT_IS_LIST
            index: Which item to extract (comes as single-element list)
            zero_indexed: If True, 0 is first item (comes as single-element list)
            
        Returns:
            tuple: (selected_item, list_length)
        """
        # When INPUT_IS_LIST = True, all inputs come wrapped in lists
        # Unwrap scalar inputs
        if isinstance(index, list):
            index = index[0] if index else 0
        if isinstance(zero_indexed, list):
            zero_indexed = zero_indexed[0] if zero_indexed else True
        
        # list_input is already a list from OUTPUT_IS_LIST, no unwrapping needed
        
        # Adjust index if 1-indexed
        actual_index = index if zero_indexed else index - 1
        
        # Validate index
        if actual_index < 0 or actual_index >= len(list_input):
            return (None, len(list_input))
        
        selected = list_input[actual_index]
        length = len(list_input)
        
        return (selected, length)