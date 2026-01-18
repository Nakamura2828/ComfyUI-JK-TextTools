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
from .detection_to_bbox import DetectionToBBox
from .bbox_to_mask import BBoxToMask
from .bboxes_to_mask import BBoxesToMask
from .json_to_bbox import JSONToBBox
from .segs_to_mask import SEGsToMask
from .bbox_to_sam3_query import BBoxToSAM3Query
from .segs_to_sam3_query import SEGsToSAM3Query
from .mask_to_bbox import MaskToBBox

NODE_CLASS_MAPPINGS = {
    "JK_StringIndexSelector": StringIndexSelector,
    "JK_StringSplitter": StringSplitter,
    "JK_ListIndexSelector": ListIndexSelector,
    "JK_StringJoiner": StringJoiner,
    "JK_JSONPrettyPrinter": JSONPrettyPrinter,
    "JK_DetectionQuery": DetectionQuery,
    "JK_DetectionToBBox": DetectionToBBox,
    "JK_BBoxToMask": BBoxToMask,
    "JK_MaskToBBox": MaskToBBox,
    "JK_BBoxesToMask": BBoxesToMask,
    "JK_JSONToBBox": JSONToBBox,
    "JK_SEGsToMask": SEGsToMask,
    "JK_BBoxToSAM3Query": BBoxToSAM3Query,
    "JK_SEGsToSAM3Query": SEGsToSAM3Query
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JK_StringIndexSelector": "String Index Selector",
    "JK_StringSplitter": "String Splitter",
    "JK_ListIndexSelector": "List Index Selector",
    "JK_StringJoiner": "String Joiner",
    "JK_JSONPrettyPrinter": "JSON Pretty Printer",
    "JK_DetectionQuery": "Query Detection JSON",
    "JK_DetectionToBBox": "Detection to BBOX",
    "JK_BBoxToMask": "BBOX to Mask",
    "JK_MaskToBBox": "Mask to BBox",
    "JK_BBoxesToMask": "BBOXes to Unified Mask",
    "JK_JSONToBBox": "JSON to BBOX",
    "JK_SEGsToMask": "SEGs to Mask",
    "JK_BBoxToSAM3Query": "BBox to SAM3 Query",
    "JK_SEGsToSAM3Query": "SEGS to SAM3 Query"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✓ JK-TextTools loaded successfully")
