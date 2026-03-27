"""
Decode the hex bytes from the C source file and display as a hex dump with ASCII.
This lets you SEE the strings hidden in your model (like "TFL3", "dense_1", etc.)

Usage:
    python decode_model_bytes.py
    python decode_model_bytes.py --start 0 --end 200      # show only bytes 0-199
    python decode_model_bytes.py --find "TFL3"             # find a string in the model
"""

import re
import argparse
from pathlib import Path


def parse_hex_from_c_file(filepath: str) -> list[int]:
    """Read a .c file and extract all hex bytes (0x__) from the array."""
    text = Path(filepath).read_text()

    # Find all hex values like 0x1c, 0xff, 0x00, etc.
    hex_pattern = re.compile(r"0x([0-9a-fA-F]{2})")
    matches = hex_pattern.findall(text)

    # Convert each hex string to an integer (0-255)
    return [int(h, 16) for h in matches]


def hex_dump(data: list[int], start: int = 0, end: int | None = None) -> None:
    """Print bytes in classic hex dump format: offset | hex bytes | ASCII."""
    if end is None:
        end = len(data)

    # Clamp to valid range
    start = max(0, start)
    end = min(len(data), end)

    print(f"\nShowing bytes {start} to {end - 1} (total: {end - start} bytes)")
    print(f"{'Offset':<10} {'Hex Bytes':<52} {'ASCII'}")
    print("-" * 80)

    for i in range(start, end, 12):
        # Get up to 12 bytes for this row
        row_bytes = data[i : min(i + 12, end)]

        # Format hex part: "1c 00 00 00 54 46 4c 33"
        hex_part = " ".join(f"{b:02x}" for b in row_bytes)

        # Format ASCII part: printable chars shown, others replaced with '.'
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in row_bytes)

        print(f"{i:08x}  {hex_part:<48} |{ascii_part}|")


def find_string(data: list[int], search: str) -> None:
    """Find all occurrences of an ASCII string in the byte data."""
    search_bytes = [ord(c) for c in search]
    found = False

    for i in range(len(data) - len(search_bytes) + 1):
        if data[i : i + len(search_bytes)] == search_bytes:
            print(f"\n  Found '{search}' at byte offset {i} (0x{i:04x})")
            # Show context: 16 bytes before and after
            context_start = max(0, i - 8)
            context_end = min(len(data), i + len(search_bytes) + 8)
            hex_dump(data, context_start, context_end)
            found = True

    if not found:
        print(f"\n  '{search}' not found in model bytes.")


def find_all_strings(data: list[int], min_length: int = 4) -> None:
    """Find all printable ASCII strings of at least min_length characters."""
    print(f"\nAll ASCII strings (length >= {min_length}):")
    print("-" * 60)

    current_string = ""
    string_start = 0

    for i, b in enumerate(data):
        if 32 <= b < 127:  # printable ASCII
            if not current_string:
                string_start = i
            current_string += chr(b)
        else:
            if len(current_string) >= min_length:
                print(f"  Offset {string_start:5d} (0x{string_start:04x}): \"{current_string}\"")
            current_string = ""

    # Don't forget the last string if file ends with printable chars
    if len(current_string) >= min_length:
        print(f"  Offset {string_start:5d} (0x{string_start:04x}): \"{current_string}\"")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode hex bytes from C model file")
    parser.add_argument(
        "--file",
        type=str,
        default="magic_wand_model_model_data.c",
        help="Path to the .c file containing the hex array",
    )
    parser.add_argument("--start", type=int, default=0, help="Start byte offset")
    parser.add_argument("--end", type=int, default=None, help="End byte offset")
    parser.add_argument("--find", type=str, default=None, help="Find a specific ASCII string")
    parser.add_argument(
        "--strings", action="store_true", help="Show all ASCII strings found in the model"
    )
    args = parser.parse_args()

    # Parse hex bytes from the C file
    print(f"Reading: {args.file}")
    data = parse_hex_from_c_file(args.file)
    print(f"Parsed {len(data)} bytes from C array")

    if args.find:
        # Search mode: find a specific string
        find_string(data, args.find)
    elif args.strings:
        # String discovery mode: find ALL strings
        find_all_strings(data)
    else:
        # Default: hex dump mode
        # If no range specified, show first 128 bytes (enough to see header + TFL3)
        if args.end is None and args.start == 0:
            args.end = 128
            print("(Showing first 128 bytes. Use --end to see more, or --strings to find text)")
        hex_dump(data, args.start, args.end)


if __name__ == "__main__":
    main()
