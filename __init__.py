"""
ComfyUI-JK-TextTools
Text and data manipulation nodes for ComfyUI
"""

from .string_index_selector import StringIndexSelector

NODE_CLASS_MAPPINGS = {
    "JK_StringIndexSelector": StringIndexSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JK_StringIndexSelector": "String Index Selector",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✓ JK-TextTools loaded successfully")
