#!/bin/bash
# This script converts a TFLite model file to a C source file.

# Stop on first error
set -e

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_model.tflite>"
    exit 1
fi

# Check if the file exists
if [ ! -f "$1" ]; then
    echo "Error: File not found at '$1'"
    exit 1
fi

# Get the base name of the model file, e.g., "magic_wand_model"
BASENAME=$(basename "$1" .tflite)
# Define the output C file name
OUT_C_FILE="${BASENAME}_model_data.c"

echo "Converting $1 to $OUT_C_FILE..."

# Convert the binary file to a C source file.
# We pass the filename directly to xxd (no '<') so it generates the
# 'unsigned char' array definition for us.
xxd -i "$1" > "$OUT_C_FILE"

echo "Done. Now, manually edit '$OUT_C_FILE' to add 'const' and 'alignas(16)'.
