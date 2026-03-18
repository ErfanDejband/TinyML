# Step 2: Convert TFLite Model → C Array

## What This Step Does

Your `.tflite` file is a binary blob. Microcontrollers have no filesystem —
so we bake the model directly into firmware as a C byte array.

```
┌─────────────────┐         ┌──────────────────────────┐
│  model.tflite   │  ────►  │  model.c + model.h       │
│  (binary file)  │  xxd    │  (C array of hex bytes)  │
└─────────────────┘         └──────────────────────────┘
```

## Why Bash + xxd (not Python)?

| Reason | Explanation |
|--------|-------------|
| **Industry standard** | TFLite Micro docs recommend `xxd -i` |
| **No dependencies** | `xxd` ships with Linux/macOS/Git Bash |
| **Clean output** | Real C files, not Python string-soup |
| **Reproducible** | One command, same result every time |

## How to Run

```bash
cd Phase_2
bash convert_tflite_to_c.sh ../models/wave_model_quantized.tflite ./c_model
```

Output:
- `c_model/wave_model_quantized_model.h` — header to `#include`
- `c_model/wave_model_quantized_model.c` — the byte array

## What xxd -i Actually Does

`xxd -i` reads any binary and outputs hex bytes in C format:

```
Input (binary):  [0x1A] [0x00] [0x00] [0x00] [0x14] ...
Output (text):   0x1a, 0x00, 0x00, 0x00, 0x14, ...
```

We just wrap those bytes in a proper C array with a header guard.

## Why alignas(16)?

Many MCUs (ARM Cortex-M, ESP32) access memory fastest when
data starts at an address divisible by 16:

```
Memory addresses:
0x00  0x04  0x08  0x0C  0x10  0x14  0x18  0x1C  0x20
                          ▲
                     alignas(16) puts your array HERE
                     (0x10 = 16, divisible by 16 ✓)
```

Without alignment → extra clock cycles to fetch data spanning two memory chunks.

## Why .h + .c (not one file)?

```
❌ The WRONG way (everything in .h):
  main.c    includes model.h  →  copy 1 of 12KB
  infer.c   includes model.h  →  copy 2 of 12KB
  Total: 24KB wasted!

✅ The RIGHT way (.h declares, .c defines):
  model.c   has the data       →  one 12KB copy
  main.c    includes model.h   →  just knows WHERE
  infer.c   includes model.h   →  just knows WHERE
  Total: 12KB ✓
```

The `extern` keyword in the header says: "this exists somewhere, the linker will find it."

## What the Output Looks Like

**Header (.h):**
```c
#ifndef WAVE_MODEL_QUANTIZED_MODEL_H
#define WAVE_MODEL_QUANTIZED_MODEL_H
#include <stdalign.h>

extern alignas(16) const unsigned char wave_model_quantized_model_data[];
extern const unsigned int wave_model_quantized_model_data_len;

#endif
```

**Source (.c):**
```c
#include "wave_model_quantized_model.h"

alignas(16) const unsigned char wave_model_quantized_model_data[] = {
    0x1a, 0x00, 0x00, 0x00, 0x14, 0x00, ...
};

const unsigned int wave_model_quantized_model_data_len = 12345;
```

## Next Step

→ **Step 3**: Write inference code in C using TFLite Micro runtime
