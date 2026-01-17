"""
ComfyUI-JK-TextTools
Text and data manipulation nodes for ComfyUI
"""

from .string_index_selector import StringIndexSelector
from .string_splitter import StringSplitter
from .list_index_selector import ListIndexSelector
from .string_joiner import StringJoiner
from .json_pretty_printer import JSONPrettyPrinter
from .detection_query import DetectionQuery

NODE_CLASS_MAPPINGS = {
    "JK_StringIndexSelector": StringIndexSelector,
    "JK_StringSplitter": StringSplitter,
    "JK_ListIndexSelector": ListIndexSelector,
    "JK_StringJoiner": StringJoiner,
    "JK_JSONPrettyPrinter": JSONPrettyPrinter,
    "JK_DetectionQuery": DetectionQuery,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JK_StringIndexSelector": "String Index Selector",
    "JK_StringSplitter": "String Splitter",
    "JK_ListIndexSelector": "List Index Selector",
    "JK_StringJoiner": "String Joiner",
    "JK_JSONPrettyPrinter": "JSON Pretty Printer",
    "JK_DetectionQuery": "Query Detection JSON Query"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✓ JK-TextTools loaded successfully")
