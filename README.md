# ComfyUI-JK-TextTools

Text and data manipulation nodes for ComfyUI.

## Features

### String Index Selector
Extract a single element from a delimited string by index.

**Use case:** In a loop, get the Nth item from a comma-separated list.

**Example:**
- Input: `"10,25,42,100"`, index: `2`
- Output: `"42"`

Perfect for extracting frame numbers, filenames, or any list item during iteration.

### String Splitter
*(Coming soon)* Split delimited strings into multiple outputs or a list.

## Installation

### Method 1: ComfyUI Manager (Recommended when published)
Search for "JK-TextTools" in ComfyUI Manager

### Method 2: Manual Installation

1. Navigate to your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/Nakamura2828/ComfyUI-JK-TextTools.git
   ```

3. Restart ComfyUI

## Nodes

### String Index Selector

**Category:** `text/string_ops`

**Inputs:**
- `text` (STRING): The delimited string to split
- `delimiter` (STRING): Character(s) to split on (default: `,`)
- `index` (INT): Which item to extract (0-based by default)
- `strip_whitespace` (BOOLEAN): Remove leading/trailing spaces (default: `True`)
- `zero_indexed` (BOOLEAN, optional): Use 0-based indexing (default: `True`)

**Outputs:**
- `selected_item` (STRING): The extracted item
- `item_count` (INT): Total number of items in the list

**Features:**
- Handles out-of-range indices gracefully (returns empty string)
- Option for 0-based or 1-based indexing
- Returns total count for validation

## Use Cases

### Loop Over Frame Numbers
```
VHS "Select Images" Node → "10,25,42,100"
    ↓
Easy Use For Loop Nodes → iteration index
    ↓
String Index Selector → "42"
    ↓
Save Image: "frame_42.png"
```

## Development

### Running Tests
```bash
python test_string_index_selector.py
```

### Requirements
- Python 3.10+
- ComfyUI

## Roadmap

- [x] String Index Selector
- [x] String Splitter (delimited string to list of string, int, or float, supports \n and \t as delimiters)
- [x] String Joiner (supports \n and \t as delimiters)
- [ ] JSON Parser
- [ ] JSON Builder
- [ ] CSV Parser

## License

MIT License

## Author

John Knox (Nakamura2828)

## Contributing

Issues and pull requests welcome!
