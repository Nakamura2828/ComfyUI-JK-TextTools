"""
ComfyUI-JK-TextTools
Text and data manipulation nodes for ComfyUI
"""

from .string_index_selector import StringIndexSelector
from .string_splitter import StringSplitter
from .list_index_selector import ListIndexSelector

NODE_CLASS_MAPPINGS = {
    "JK_StringIndexSelector": StringIndexSelector,
    "JK_StringSplitter": StringSplitter,
    "JK_ListIndexSelector": ListIndexSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JK_StringIndexSelector": "String Index Selector",
    "JK_StringSplitter": "String Splitter",
    "JK_ListIndexSelector": "List Index Selector",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✓ JK-TextTools loaded successfully")
