#!/bin/bash
# ============================================================
# Convert a .tflite model to C source + header
# Usage: bash convert_tflite_to_c.sh <model.tflite> [output_dir]
# ============================================================

set -e

# --- Input validation ---
if [ -z "$1" ]; then
    echo "Usage: bash convert_tflite_to_c.sh <model.tflite> [output_dir]"
    echo "Example: bash convert_tflite_to_c.sh ../models/wave_model_quantized.tflite ./c_model"
    exit 1
fi

TFLITE_FILE="$1"
OUTPUT_DIR="${2:-.}"

if [ ! -f "$TFLITE_FILE" ]; then
    echo "Error: File '$TFLITE_FILE' not found!"
    exit 1
fi

# --- Setup names ---
BASENAME=$(basename "$TFLITE_FILE" .tflite)
ARRAY_NAME=$(echo "${BASENAME}_model" | tr '-.' '_')
GUARD_NAME=$(echo "${ARRAY_NAME}_H" | tr '[:lower:]' '[:upper:]')
MODEL_SIZE=$(wc -c < "$TFLITE_FILE" | tr -d ' ')

mkdir -p "$OUTPUT_DIR"

echo "Converting: $TFLITE_FILE"
echo "Model size: $MODEL_SIZE bytes ($(echo "scale=2; $MODEL_SIZE / 1024" | bc) KB)"
echo "Array name: $ARRAY_NAME"
echo "Output dir: $OUTPUT_DIR"

# --- Generate .c file ---
cat > "$OUTPUT_DIR/${ARRAY_NAME}.c" << EOF
// ${ARRAY_NAME}.c
// Auto-generated from $(basename "$TFLITE_FILE")
// Model size: ${MODEL_SIZE} bytes
//
// DO NOT EDIT — regenerate with convert_tflite_to_c.sh

#include "${ARRAY_NAME}.h"

alignas(16) const unsigned char ${ARRAY_NAME}_data[] = {
EOF

# xxd -i reads binary, outputs C hex bytes like: 0x1a, 0x00, 0x00, ...
xxd -i < "$TFLITE_FILE" | sed 's/^/    /' >> "$OUTPUT_DIR/${ARRAY_NAME}.c"

cat >> "$OUTPUT_DIR/${ARRAY_NAME}.c" << EOF
};

const unsigned int ${ARRAY_NAME}_data_len = ${MODEL_SIZE};
EOF

# --- Generate .h file ---
cat > "$OUTPUT_DIR/${ARRAY_NAME}.h" << EOF
// ${ARRAY_NAME}.h
// Auto-generated from $(basename "$TFLITE_FILE")
// Model size: ${MODEL_SIZE} bytes
//
// DO NOT EDIT — regenerate with convert_tflite_to_c.sh

#ifndef ${GUARD_NAME}
#define ${GUARD_NAME}

#include <stdalign.h>

extern alignas(16) const unsigned char ${ARRAY_NAME}_data[];
extern const unsigned int ${ARRAY_NAME}_data_len;

#endif // ${GUARD_NAME}
EOF

echo ""
echo "Done! Generated:"
echo "  $OUTPUT_DIR/${ARRAY_NAME}.h"
echo "  $OUTPUT_DIR/${ARRAY_NAME}.c"
echo ""
echo "Usage in your C code:"
echo "  #include \"${ARRAY_NAME}.h\""
echo "  // bytes: ${ARRAY_NAME}_data"
echo "  // size:  ${ARRAY_NAME}_data_len"
