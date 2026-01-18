"""
Master test runner for ComfyUI-JK-TextTools

Run all tests at once:
    python tests/run_all_tests.py

Or from project root:
    python -m tests.run_all_tests
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import node modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test modules
from tests import (
    test_string_index_selector,
    test_string_splitter,
    test_list_index_selector,
    test_string_joiner,
    test_json_pretty_printer,
    test_detection_query,
    test_detection_to_bbox,
    test_bbox_to_mask,
    test_json_to_bbox,
    test_segs_to_mask
)


def run_all_tests():
    """Run all test suites"""
    print("="*60)
    print("Running ALL tests for ComfyUI-JK-TextTools")
    print("="*60)
    print()
    
    test_modules = [
        ("String Index Selector", test_string_index_selector),
        ("String Splitter", test_string_splitter),
        ("List Index Selector", test_list_index_selector),
        ("String Joiner", test_string_joiner),
        ("JSON Pretty Printer", test_json_pretty_printer),
        ("Detection Query", test_detection_query),
        ("Detection to BBox", test_detection_to_bbox),
        ("BBox to Mask", test_bbox_to_mask),
        ("JSON to BBox", test_json_to_bbox),
        ("SEGs to Mask", test_segs_to_mask),
    ]
    
    results = []
    
    for name, module in test_modules:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        success = module.run_all_tests()
        results.append((name, success))
        print()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
        if not success:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)