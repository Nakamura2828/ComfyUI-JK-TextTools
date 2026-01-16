# Project Context for Claude

## Project Overview
ComfyUI-JK-TextTools is a custom node package for ComfyUI that provides text and data manipulation utilities.

**Current Status:** Working but needs refinement
- All nodes functional and tested
- String Index Selector and String Splitter work together
- List handling works within our ecosystem
- OUTPUT_IS_LIST feature not displaying correctly (grid icon issue)

**Primary Goal:** Provide text manipulation tools for ComfyUI workflows, particularly for loop-based processing

## Project Structure
```
ComfyUI-JK-TextTools/
├── __init__.py                      # Node registration
├── string_index_selector.py         # Extract item from delimited string by index
├── string_splitter.py               # Split delimited string into list
├── list_index_selector.py           # Extract item from list by index
├── test_string_index_selector.py    # Tests for string index selector
├── test_string_splitter.py          # Tests for string splitter
├── test_list_index_selector.py      # Tests for list index selector
├── README.md                        # Documentation
├── requirements.txt                 # Development dependencies
└── .gitignore                       # Git ignore rules
```

## Development Environment
- **Python Version:** 3.13 (matching ComfyUI's embedded Python)
- **Environment:** Anaconda conda environment (comfyui-dev)
- **IDE:** VS Code with Python, Pylance, GitLens extensions
- **ComfyUI Location:** `C:\ComfyUI_windows_portable\ComfyUI`
- **Dev Directory:** `C:\Users\jknox\Projects\ComfyUI-JK-TextTools`
- **Symlink:** From ComfyUI's custom_nodes to dev directory for testing

## Current Nodes

### 1. String Index Selector
**Purpose:** Extract a single element from a delimited string by index (perfect for loops)

**Status:** ✅ Working perfectly

**Key Features:**
- Direct operation on delimited strings (no intermediate nodes needed)
- Configurable delimiter
- Zero-based or one-based indexing
- Whitespace stripping
- Graceful out-of-range handling
- Returns item count for validation

**Use Case:** Original motivation - extracting frame numbers from comma-delimited string in loop workflows

### 2. String Splitter
**Purpose:** Split delimited string into a list of strings

**Status:** ✅ Functional but OUTPUT_IS_LIST not working as expected

**Current Issue:** 
- Using `OUTPUT_IS_LIST = (True, False)` to mark first output as list
- ComfyUI shows normal circle icon instead of grid icon (batch/list indicator)
- List is created correctly and works with our List Index Selector
- ImpactPack's equivalent node shows grid icon, proving feature works in ComfyUI

**Key Features:**
- Multiple delimiter support
- Whitespace stripping
- Empty string removal option
- Returns item count

### 3. List Index Selector
**Purpose:** Extract item from a list by index

**Status:** ✅ Works with String Splitter output

**Compatibility Notes:**
- Works perfectly with our String Splitter
- Accepts ImpactPack's list output but treats it as string (gets characters, not list items)
- This suggests different list implementation approaches between packages

**Current Implementation:**
- Input type: `"*"` (wildcard) with `forceInput: True`
- No `INPUT_IS_LIST` (removed during debugging)
- Accepts any connected input

## Known Issues

### Issue #1: OUTPUT_IS_LIST Not Displaying Grid Icon
**Symptom:** String Splitter output shows circle icon instead of grid icon

**What We've Tried:**
- Using `OUTPUT_IS_LIST = (True, False)`
- Various input/output type configurations
- Reviewing ComfyUI documentation on list handling

**What Works Elsewhere:**
- ImpactPack's "Make List" node shows grid icon correctly
- This proves OUTPUT_IS_LIST works in our ComfyUI version

**Next Steps to Debug:**
1. Check exact ImpactPack source code implementation
2. Verify ComfyUI version and OUTPUT_IS_LIST support
3. Try minimal test node with just OUTPUT_IS_LIST
4. Check ComfyUI console for warnings during node load

### Issue #2: Cross-Package List Compatibility
**Symptom:** ImpactPack's list passed to our List Index Selector treats it as string

**Analysis:** Different packages use different list mechanisms
- Our approach: Python list with OUTPUT_IS_LIST
- ImpactPack approach: Unknown (needs investigation)

**Current Stance:** Documented limitation, not critical for primary use cases

## Original Use Case (Frame Number Extraction)
User had workflow with:
- VHS node outputting comma-delimited frame numbers: `"10,25,42,100"`
- For loop iterating with index
- Needed to extract corresponding frame number for each iteration
- Worked around with regex→newlines + multiline text selector

**Our Solution:**
String Index Selector directly solves this:
```
"10,25,42,100" + index 2 → "42"
```

No intermediate nodes needed!

## Testing Approach
- Comprehensive unit tests for all nodes
- Tests verify function logic, not ComfyUI integration
- All tests passing ✅
- Manual testing in ComfyUI for UI verification

## Git Workflow
- Repository: `https://github.com/Nakamura2828/ComfyUI-JK-TextTools`
- Development on main branch
- Testing via symlink to ComfyUI custom_nodes
- Regular commits with descriptive messages

## Development Preferences
- Write tests first or alongside implementation
- Include validation tests (INPUT_TYPES matches function signature)
- Follow Python naming conventions (snake_case)
- Comprehensive error handling
- Clear documentation in docstrings

## Future Roadmap
- [ ] Debug OUTPUT_IS_LIST grid icon issue
- [ ] Improve cross-package list compatibility (if feasible)
- [ ] String Joiner node
- [ ] JSON Parser node
- [ ] JSON Builder node
- [ ] CSV Parser node

## Important Context for Debugging

### ComfyUI List System (from documentation)
- `OUTPUT_IS_LIST`: Tuple of booleans matching RETURN_TYPES length
- `INPUT_IS_LIST`: Boolean or tuple specifying which inputs accept lists
- ComfyUI wraps/unwraps lists automatically
- Grid icon indicates list/batch type in UI

### What We Know Works
1. String Index Selector + delimited string (primary use case) ✅
2. String Splitter → List Index Selector (within our package) ✅
3. All unit tests ✅
4. ImpactPack's list system (proves feature exists) ✅

### What Doesn't Work Yet
1. OUTPUT_IS_LIST not showing grid icon ❌
2. Cross-compatibility with ImpactPack lists ❌

## Questions to Investigate
1. What exact ComfyUI version are we running?
2. When was OUTPUT_IS_LIST added to ComfyUI?
3. How exactly does ImpactPack implement their list nodes?
4. Is there a ComfyUI version requirement we're missing?
5. Are we wrapping the return value correctly for OUTPUT_IS_LIST?

## Notes for Claude Code
- When debugging, check ComfyUI console output during node loading
- ComfyUI may silently ignore OUTPUT_IS_LIST if syntax is wrong
- Test changes require ComfyUI restart
- Symlink means changes in dev directory immediately affect ComfyUI
- Check ImpactPack source at: `C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Impact-Pack\`

## Learning Goals
This project is also a learning experience for:
- ComfyUI custom node development
- Proper testing practices
- Git workflow
- VS Code development environment
- Python best practices

The user values understanding *why* things work, not just making them work.
