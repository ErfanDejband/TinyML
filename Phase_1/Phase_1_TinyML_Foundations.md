# 🧠 Phase 1: TinyML Foundations — Complete Learning Guide

> **What this document covers:** Everything from raw sensor data to an optimized `.tflite` model.  
> **Prerequisites:** Basic Python, basic understanding of what a neural network is.  
> **Outcome:** You will deeply understand sliding windows, CNNs for time-series, pruning mechanics, quantization math, and the TFLite conversion pipeline.

---

## Table of Contents
1. [The Big Picture — What Is TinyML?](#1-the-big-picture)
2. [Data Collection — Phone Sensors](#2-data-collection)
3. [Data Preprocessing — The Sliding Window](#3-data-preprocessing)
4. [The Model — Conv1D CNN](#4-the-model)
5. [The Training Loop — Steps, Batches, Epochs](#5-the-training-loop)
6. [Pruning — Removing Weak Connections](#6-pruning)
7. [Quantization — Shrinking Every Number](#7-quantization)
8. [The TFLite Converter — From Python to Binary](#8-the-tflite-converter)
9. [The tf_keras Problem — Why It Exists](#9-the-tf-keras-problem)
10. [Tools & Next Steps](#10-tools-and-next-steps)
11. [Self-Test Quiz](#11-self-test-quiz)

---

## 1. The Big Picture

### What Is TinyML?
TinyML = **Machine Learning on microcontrollers** (devices with kilobytes of RAM, not gigabytes).

```
Traditional ML:                          TinyML:
┌─────────────┐                         ┌─────────────┐
│  GPU Server │                         │  Arduino    │
│  16 GB RAM  │                         │  256 KB RAM │
│  200W power │                         │  0.1W power │
│  Cloud-based│                         │  On-device  │
└─────────────┘                         └─────────────┘
   Model: 500 MB                           Model: 5 KB
   Latency: 100ms + network               Latency: 10ms
   Privacy: Data sent to cloud            Privacy: Data stays on device
```

### The Pipeline (What You Built)
```
Phone Sensor → JSON → CSV → Sliding Windows → Train CNN → Prune → Quantize → .tflite → C Header → Hardware
     ↑                                                                                              ↑
  Phase 1 START                                                                              Phase 1 END
```

### The Sculpting Analogy
Think of the entire Phase 1 as sculpting:
1. **Training** = Creating the rough clay shape (the "dense" model)
2. **Pruning** = Cutting away unnecessary clay to make it lighter
3. **Quantization** = Baking the clay into hard, compact ceramic (float32 → int8)

You **cannot** prune or quantize something that doesn't exist yet. Always train first.

---

## 2. Data Collection

### Source
- **Device**: Smartphone accelerometer via Edge Impulse web interface
- **Sensor**: 3-axis accelerometer (x, y, z)
- **Sampling Rate**: ~50Hz (one reading every ~20ms)
- **Gestures Recorded**: "Wave" (side to side) and "Idle" (phone on table)
- **Duration**: ~2 minutes per gesture

### Raw Data Format (JSON from phone)
```json
{
  "payload": {
    "sensors": [
      {"name": "accX", "units": "m/s2"},
      {"name": "accY", "units": "m/s2"},
      {"name": "accZ", "units": "m/s2"}
    ],
    "interval_ms": 20,
    "values": [
      [0.0, 0.2, 9.7],
      [0.1, 0.3, 9.8],
      ...
    ]
  }
}
```

### What `process_data.py` Does
- Reads JSON files from the `RowData/` folder
- Extracts `interval_ms` to reconstruct timestamps
- Builds a clean DataFrame: `time, x, y, z, label`
- The `label` comes from the filename (e.g., `wave_001.json` → "Wave")

---

## 3. Data Preprocessing — The Sliding Window

### The Problem
A single accelerometer reading `(0.0, 0.2, 9.7)` tells you **nothing** about motion.

> 🧠 **Analogy**: One frame of a movie doesn't tell you the story. You need 30 frames (1 second) to see the action.

### The Solution: Stack Readings Into Windows
```
Raw data (point by point):          Windowed data (grouped):
┌───────────────────────┐          ┌──────────────────────────────┐
│ t=0.00: 0.0, 0.2, 9.7 │          │ Window 1: rows 0-49  (1 sec) │ → Label: "Idle"
│ t=0.02: 0.1, 0.3, 9.8 │          │ Window 2: rows 25-74 (1 sec) │ → Label: "Idle"
│ t=0.04: 0.0, 0.2, 9.7 │          │ Window 3: rows 50-99 (1 sec) │ → Label: "Wave"
│ ...                   │          │ ...                          │
│ t=20.0: 0.5, 0.1, 9.6 │          │ Window N: last 50 rows       │ → Label: "Wave"
└───────────────────────┘          └──────────────────────────────┘
  Shape: (1000, 3)                    Shape: (N, 50, 3)
  One number per row                  One MATRIX per sample
```

### The Math
| Parameter | Value | Why |
|-----------|-------|-----|
| **Sampling Rate** | 50Hz | Phone sends 50 readings/second |
| **Gesture Duration** | ~1 second | A "Wave" takes about 1 second |
| **Window Size** | 50 samples | 50Hz × 1s = 50 readings |
| **Step Size (Stride)** | 25 samples | 50% overlap for data augmentation |
| **Input Shape** | `(50, 3)` | 50 timesteps × 3 axes (x, y, z) |

### Step Size / Overlap Explained
```
No Overlap (step = window = 50):
|████████████████████|                    |████████████████████|
     Window 1                                  Window 2
                      ← GAP! Gesture here is MISSED

50% Overlap (step = 25):
|████████████████████|
          |████████████████████|
                    |████████████████████|
   Window 1     Window 2     Window 3
                ← No gaps! Every gesture position is captured
```

### The Label Assignment
Each window gets ONE label. How? Take the **mode** (most common) label within those 50 rows.
- If 48 rows say "Wave" and 2 say "Idle" → Window label = "Wave"

### What `prepare_data_for_training.py` Does
1. Calls `process_data.py` to get the DataFrame
2. Creates sliding windows with configurable size and overlap
3. Encodes labels: `LabelEncoder()` → "Idle" = 0, "Wave" = 1
4. Splits data: `train_test_split()` with `stratify` (keeps label ratio equal in both sets)
5. Output: `X_train (478, 50, 3)`, `X_test (120, 50, 3)`, `y_train (478,)`, `y_test (120,)`

---

## 4. The Model — Conv1D CNN

### Why CNN for Time-Series?
A **Conv1D** layer slides a small filter across time, detecting local patterns:
```
Input signal (50 timesteps of accelerometer x):
  ╭─╮   ╭─╮   ╭─╮
──╯ ╰───╯ ╰───╯ ╰──   ← "Wave" pattern (oscillation)

Conv1D filter (size 3) slides across:
  [w1, w2, w3] →→→→→→   ← Detects "up-down-up" shapes
```

### Your Architecture
```
Input: (50, 3)
    ↓
Conv1D(filters=8, kernel_size=3, activation='relu')
    ↓  Output: (48, 8)  ← 8 different "pattern detectors"
MaxPooling1D(pool_size=2)
    ↓  Output: (24, 8)  ← Halves the time dimension
Flatten()
    ↓  Output: (192,)   ← 24 × 8 = 192 flat features
Dense(16, activation='relu')
    ↓  Output: (16,)     ← Compress to 16 features
Dense(2, activation='softmax')
    ↓  Output: (2,)      ← [P(Idle), P(Wave)]
```

### Parameter Count
| Layer | Parameters | Calculation |
|-------|-----------|-------------|
| Conv1D | 80 | (3 kernel × 3 features × 8 filters) + 8 biases |
| Dense(16) | 3,088 | (192 inputs × 16 units) + 16 biases |
| Dense(2) | 34 | (16 inputs × 2 units) + 2 biases |
| **Total** | **3,202** | **~12.51 KB as float32** |

### Why `softmax`?
The output `[0.15, 0.85]` means: "15% chance of Idle, 85% chance of Wave."  
All values sum to 1.0 — it's a probability distribution.

### What `create_model.py` Does
- Builds a `keras.Sequential` model with configurable filters, kernel size, dense layers
- Compiles with Adam optimizer and sparse categorical crossentropy loss
- Returns the compiled model ready for `.fit()`

---

## 5. The Training Loop — Steps, Batches, Epochs

### Definitions
| Term | What It Is | Analogy |
|------|-----------|---------|
| **Sample** | One window of `(50, 3)` data | One flashcard |
| **Batch** | A group of samples processed together (default: 32) | A stack of 32 flashcards |
| **Step** | Processing one batch through the network | Studying one stack |
| **Epoch** | One complete pass through ALL batches | Going through the entire deck |

### Your Numbers
```
Total samples: 478
Batch size: 32 (default)
Steps per epoch: ceil(478 / 32) = ceil(14.9375) = 15
                                                    ↑
                                    14 full batches of 32 + 1 partial batch of 30

Training output:
Epoch 1/30
15/15 [==============================] - 1s 15ms/step
 ↑  ↑
 │  └── Total steps this epoch
 └───── Current step
```

### Why 15, Not 14?
Keras uses **ceiling division**. It won't throw away the last 30 samples.
- 14 batches × 32 = 448 samples processed
- 1 batch × 30 = remaining 30 samples  
- Total: 478 ✅ (all data used)

### Why This Matters for Pruning
The pruning schedule uses **global steps**, not epochs:
```
Total global steps = steps_per_epoch × epochs = 15 × 20 = 300

If you set end_step=1000 but only train for 300 steps:
  → You'll only reach ~10% sparsity instead of 50%!
  → Your model stays "fat" and you won't know why.

ALWAYS calculate: end_step = steps_per_epoch × epochs
```

---

## 6. Pruning — Removing Weak Connections

### The Core Idea
Many weights in a trained model are very close to zero (e.g., `0.00001`). These weights barely affect the output. **Pruning sets them to exactly zero and freezes them.**

> 🧠 **Analogy**: A company with 100 employees where 50 barely contribute. Pruning = laying off the 50 least productive workers, then giving the remaining 50 time to pick up the slack.

### What Happens Mathematically
```
Before pruning:    W = [0.9, 0.01, 0.7, -0.03, 0.85, 0.02]
                         ↑    ↑     ↑     ↑      ↑     ↑
                        big  tiny  big   tiny   big   tiny

Sorted by |magnitude|: [0.01, 0.02, -0.03, 0.7, 0.85, 0.9]
                         ↑     ↑      ↑
                         Bottom 50% → KILL THESE

After 50% pruning:  W = [0.9, 0.00, 0.7, 0.00, 0.85, 0.00]
                               ↑           ↑            ↑
                             DEAD         DEAD         DEAD
```

### The Pruning Mask
The mask is a **shadow tensor** of the same shape as the weights, containing only `1`s and `0`s:

```
Weights: [0.9,  0.01, 0.7, -0.03, 0.85, 0.02]
Mask:    [  1,    0,    1,    0,     1,    0  ]

Effective weight = Weight × Mask
         = [0.9,  0.0,  0.7,  0.0,  0.85, 0.0]
```

During training, even if backpropagation tries to update a pruned weight, the mask **forces it back to zero** at the end of every step.

### The `UpdatePruningStep()` Callback

This callback is **mandatory**. Without it, **zero pruning occurs** (0% of weights are removed).

What it does at each training step:
```
Step 1: Normal forward pass + backpropagation (weights update)
Step 2: Callback fires:
        ├── Check global_step counter
        ├── Calculate target sparsity from schedule formula
        ├── If (step % frequency == 0):  ← Every 100 steps
        │     ├── Sort all weights by |magnitude|
        │     ├── Find the threshold for target %
        │     ├── Update Mask: weights below threshold → 0
        │     └── Apply Mask: force pruned weights to 0.0
        └── Always: Ensure masked weights stay at 0
```

> 🧠 **Analogy**: The callback is the **surgeon's hand** holding the scissors. The schedule tells the surgeon **how much** to cut. Without the hand, the scissors sit idle.

### Pruning Schedules — Two Options

#### Option A: `ConstantSparsity` 
```python
tfmot.sparsity.keras.ConstantSparsity(
    target_sparsity=0.5,   # Target: 50% of weights = 0
    begin_step=0,          # Start immediately
    end_step=300,          # Finish by step 300
    frequency=100          # Re-evaluate mask every 100 steps
)
```

**How it works**: From `begin_step`, it immediately targets 50%. Every `frequency` steps, it re-sorts weights and applies the mask. The target stays constant at 50%.

```
Sparsity
  0.5 |████████████████████████████████████
      |
      |
  0.0 |
      └─────────────────────────────────→ Steps
        0                              300
```

**When to use**: Simple projects, small models, when you don't need gradual ramping.

#### Option B: `PolynomialDecay` (What you used)(The "gentle" option)
```python
tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,   # Start with 0% pruning
    final_sparsity=0.5,     # End with 50% pruning
    begin_step=0,
    end_step=1000,
    power=3                 # Controls the curve shape
)
```

**The Formula**:
$$S_t = S_{final} + (S_{initial} - S_{final}) \times \left(1 - \frac{t - t_{begin}}{t_{end} - t_{begin}}\right)^{power}$$

**Worked Example** (S_i=0.0, S_f=0.5, begin=0, end=1000, power=3):

| Step | Progress (t/T) | (1 - t/T)³ | Sparsity = 0.5 + (0-0.5)×result | Weights Pruned |
|------|----------------|-------------|----------------------------------|----------------|
| 0 | 0.00 | 1.000 | 0.5 + (-0.5)(1.0) = **0.00** | 0% |
| 250 | 0.25 | 0.422 | 0.5 + (-0.5)(0.422) = **0.29** | 29% |
| 500 | 0.50 | 0.125 | 0.5 + (-0.5)(0.125) = **0.44** | 44% |
| 750 | 0.75 | 0.016 | 0.5 + (-0.5)(0.016) = **0.49** | 49% |
| 1000 | 1.00 | 0.000 | 0.5 + (-0.5)(0.0) = **0.50** | 50% |

```
Sparsity
  0.5 |                          ___________
      |                      ___/
      |                   __/
      |                __/      ← Accelerates in the middle
      |            ___/
  0.0 |___________/             ← Gentle start
      └─────────────────────────────────────→ Steps
        0        250      500      750    1000
```

**The `power` parameter controls the curve**:
```
power=1: Linear       ──────/──────  (constant rate)
power=2: Quadratic    ─────/───────  (slightly back-loaded)
power=3: Cubic        ────/────────  (gentle start, aggressive middle)
power=5: Quintic      ───/─────────  (very gentle start, very aggressive end)
```

**When to use**: Larger models, when accuracy is fragile, when you want the model to gradually adapt.

### The `frequency` Parameter
Even though the sparsity target changes every step, the **actual mask update** (sorting + zeroing) only happens every `frequency` steps.

```
frequency=100:
Step 1-99:    Target changes, but mask stays the same (model trains freely)
Step 100:     MASK UPDATE! Sort weights, apply new threshold, zero out weights
Step 101-199: Train with new mask
Step 200:     MASK UPDATE again!
...
```

**Why not update every step?**
- Sorting thousands of weights is computationally expensive
- The model needs "breathing room" to adjust between cuts
- For small models (like yours with 3,202 params), `frequency=100` is fine
- For large models, you might use `frequency=1000`

### `strip_pruning()` — The Final Step
After training with pruning is complete, the model contains:
- The actual weights (with many zeros)
- The pruning masks (the 1s and 0s)
- The pruning schedule metadata
- Extra wrapper layers (e.g., `prune_low_magnitude_conv1d`)

`strip_pruning()` does this:
```
Before strip:                          After strip:
┌────────────────────────────┐        ┌────────────────────────┐
│ prune_low_magnitude_conv1d │        │ conv1d                 │
│ ├── weights: [0.9, 0, 0.7] │   →    │ weights: [0.9, 0, 0.7] │
│ ├── mask:    [1,   0,  1 ] │        │                        │
│ └── schedule: {...}        │        │ (no mask)              │
└────────────────────────────┘        └────────────────────────┘
    Heavy (~2x model size)              Light (standard model)
```

The zeros are now **permanently baked** into the weights. The model is a standard Keras model again — but 50% of its weights are exactly `0.0`.

### Why Pruning Alone Doesn't Shrink the File
```
Original model:  [0.9, 0.3, 0.7, 0.1, 0.85, 0.6]  → 24 bytes (6 × float32)
Pruned model:    [0.9, 0.0, 0.7, 0.0, 0.85, 0.0]  → 24 bytes (SAME SIZE!)
                                                       ↑
                                         Still 6 float32 numbers!

Pruned + Compressed (ZIP/gzip):
   [0.9, 0.7, 0.85 + "zeros at positions 1,3,5"]  → ~14 bytes
                                                       ↑
                                         Compression loves patterns of zeros!
```

The real benefit comes when you combine pruning with **quantization** AND **compression** (which TFLite handles automatically).

---

## 7. Quantization — Shrinking Every Number

### The Problem: Float32 Is Wasteful
Every weight is stored as IEEE 754 float32:
```
float32 representation of 3.14159:
 Sign  Exponent       Mantissa
  0    10000000    10010010000111111011011
  └┘   └──────┘    └────────────────────┘
 1 bit  8 bits          23 bits           = 32 bits = 4 bytes

Range:     ±3.4 × 10³⁸
Precision: ~7 decimal digits
```

But neural network weights are typically between **-2.0 and +6.0** (or similar small ranges). You don't need 7 digits of precision for that!

> 🧠 **Analogy**: Using a shipping container to deliver a single pizza. A bicycle would do just fine.

### The Solution: Map Float32 → Int8
```
float32:  4 bytes per number  →  Range: ±3.4×10³⁸     (overkill)
int8:     1 byte per number   →  Range: -128 to +127   (just right)

Result: 4× smaller model!
```

### The Core Formula
To convert between real (float) values and quantized (int8) values:

$$V_{real} = S \times (Q_{int8} - Z)$$

Or rearranged to quantize a float:

$$Q_{int8} = \text{round}\left(\frac{V_{real}}{S}\right) + Z$$

Where:
- **V** = the real float value (e.g., `3.5`)
- **Q** = the quantized int8 value (e.g., `48`)
- **S** = Scale factor (how much "real value" one integer step represents)
- **Z** = Zero Point (which integer represents real `0.0`)

### Calculating Scale and Zero Point

**Given**: A layer has weights ranging from `V_min = -2.0` to `V_max = 6.0`

**Step 1: Calculate Scale (S)**
$$S = \frac{V_{max} - V_{min}}{Q_{max} - Q_{min}} = \frac{6.0 - (-2.0)}{127 - (-128)} = \frac{8.0}{255} \approx 0.03137$$

> S represents the "resolution" — each integer step = 0.03137 in real value.

**Step 2: Calculate Zero Point (Z)**
$$Z = Q_{min} - \frac{V_{min}}{S} = -128 - \frac{-2.0}{0.03137} = -128 + 63.75 \approx -64$$

> Z is the integer that represents real `0.0`. This is crucial: pruned weights (`0.0`) must map to a clean integer.

**Step 3: Convert individual weights**

| Real Value (V) | Formula: Q = round(V/S) + Z | Int8 Value (Q) |
|----------------|----------------------------|-----------------|
| -2.0 (min) | round(-2.0/0.03137) + (-64) = -64 + (-64) | **-128** ← Q_min ✅ |
| 0.0 (zero) | round(0.0/0.03137) + (-64) = 0 + (-64) | **-64** ← Zero maps cleanly ✅ |
| 3.5 | round(3.5/0.03137) + (-64) = 112 + (-64) | **48** |
| 6.0 (max) | round(6.0/0.03137) + (-64) = 191 + (-64) | **127** ← Q_max ✅ |

### De-quantization (Going Back)
To recover the float value from int8:
$$V = S \times (Q - Z) = 0.03137 \times (48 - (-64)) = 0.03137 \times 112 = 3.513$$

> Note: 3.513 ≠ 3.500 exactly. This tiny error (~0.013) is the "cost" of quantization. For neural networks, this is almost always acceptable.

### Visual: Float vs Int8 Number Line
```
Float32 (continuous — infinite points):
─────────────────────────────────────────────────
-2.0        0.0        2.0        4.0        6.0

Int8 (discrete — only 256 possible values):
─•──•──•──•──•──•──•──•──•──•──•──•──•──•──•──•─
-128 -96  -64  -32    0   32   64   96  127
  ↕        ↕         ↕        ↕        ↕
-2.0      0.0       2.0      4.0      6.0    (mapped float values)

Each • is a possible int8 value.
Any float between two dots gets ROUNDED to the nearest one.
```

### The Per-Layer Rule
Each layer gets its **own** S and Z! Why? Because different layers have different weight distributions:

```
Layer 1 (Conv1D):     weights range [-0.5, +0.8]  → S=0.0051, Z=-30
Layer 2 (Dense 16):   weights range [-2.0, +6.0]  → S=0.0314, Z=-64
Layer 3 (Dense 2):    weights range [-1.0, +1.0]  → S=0.0078, Z=0
```

The quantizer adapts to each layer's unique distribution.

### The Two Types of Numbers in a Model

| Type | What They Are | When We Know Their Range |
|------|--------------|-------------------------|
| **Weights** | Static numbers stored in the model file | **Immediately** — we can scan them |
| **Activations** | Dynamic numbers that flow between layers during inference | **Only when we run real data** |

This is why we need the **Representative Dataset**.

### The Representative Dataset Generator

```python
def representative_dataset_gen():
    for i in range(100):
        yield [X_train[i:i+1].astype(np.float32)]
```

**What happens inside the converter:**
```
Step 1: Feed sample 1 through the model
        → Layer 1 activations range: [-1.2, +3.4]
        → Layer 2 activations range: [-0.5, +2.1]

Step 2: Feed sample 2 through the model
        → Layer 1 activations range: [-1.5, +3.1]  ← New min!
        → Layer 2 activations range: [-0.3, +2.8]  ← New max!

... repeat for 100 samples ...

Step 100: Final observed ranges:
        → Layer 1: [-2.3, +4.1]  → S=0.025, Z=-92
        → Layer 2: [-0.9, +3.5]  → S=0.017, Z=-75

These S and Z values are stored IN the .tflite file alongside the weights.
```

> Without the representative dataset, the converter **cannot determine the activation ranges** and the quantized model would be garbage (random accuracy).

### Quantization Options in TFLite

| Method | Weights | Activations | Input/Output | Size | Speed | Best For |
|--------|---------|-------------|-------------|------|-------|----------|
| No quantization | float32 | float32 | float32 | 100% | Baseline | Development |
| Dynamic range | int8 | float32 | float32 | ~25% | Faster | Quick optimization |
| **Full integer** | **int8** | **int8** | **int8** | **~25%** | **Fastest** | **Microcontrollers ✅** |
| Float16 | float16 | float16 | float16 | ~50% | GPU only | Mobile GPUs |

**You chose Full Integer** — this is the best for TinyML because:
- Microcontrollers (Arduino, ESP32) **don't have floating-point hardware**
- Integer math is 2-4x faster than float math even on CPUs that support float
- Smallest possible model size

### Your Quantization Code Explained

```python
# 1. Create the converter from your pruned model
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)

# 2. Enable optimization (this activates quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 3. Provide real data so the converter can measure activation ranges
converter.representative_dataset = representative_dataset_gen

# 4. Force EVERYTHING to int8 (no float fallback allowed)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# 5. Even the model's input and output are int8
#    (your C code will feed int8 values, not floats!)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 6. Run the conversion
tflite_model = converter.convert()
```

### Why Int8 Input/Output Matters for C
When you write C code later (Phase 3), you will NOT do this:
```c
float input[150] = {0.5, 0.8, 0.2, ...};  // ❌ No floats!
```

You will do this:
```c
int8_t input[150] = {15, -42, 88, ...};    // ✅ Pre-quantized integers!
```

Your C code must quantize the raw sensor data using the model's input S and Z before feeding it in. This is a key Phase 3 skill.

---

## 8. The TFLite Converter — From Python to Binary

### What the Converter Actually Does

Your Keras model is a **living Python object** designed for training:
```
Keras Model (.h5 / .keras):
├── Architecture (layer types, connections)
├── Weights (float32 matrices)
├── Optimizer state (Adam momentum, learning rates)
├── Gradient computation logic
├── Training metadata
└── Python dependencies (numpy, keras, etc.)
```

The TFLite converter performs a **"medical autopsy"** — it keeps only what's needed for inference:
```
TFLite Model (.tflite):
├── Architecture (as a FlatBuffer schema)
├── Weights (int8 values + Scale/ZeroPoint per layer)
├── Operator definitions (Conv1D, Dense, etc.)
└── Nothing else. No optimizer, no gradients, no Python.
```

### Key Optimizations During Conversion

| Optimization | What It Does | Example |
|-------------|-------------|---------|
| **Op Fusion** | Merges consecutive operations into one | `Conv1D` + `ReLU` → single `CONV_2D` op |
| **Constant Folding** | Pre-calculates anything that can be known at compile time | Bias addition baked into weights |
| **Dead Code Elimination** | Removes unused layers/operations | Dropout (only needed during training) |
| **Format Conversion** | Restructures data layout for hardware | Channel-last → Channel-first if needed |

### The FlatBuffer Format
A `.tflite` file is a **FlatBuffer** — a binary format designed for:
- **Zero-copy access**: The C code can read the model directly from memory without parsing
- **No deserialization needed**: Unlike JSON or protobuf, no "unpacking" step
- **Memory-mapped**: On a microcontroller, the model sits in Flash ROM and is read in-place

```
.tflite file structure:
┌──────────────────────────────┐
│ Header (magic bytes: "TFL3") │  ← 4 bytes
├──────────────────────────────┤
│ Schema (table of contents)   │  ← Where each tensor starts
├──────────────────────────────┤
│ Operator list                │  ← Conv1D, Dense, etc.
├──────────────────────────────┤
│ Tensor 1: Conv1D weights     │  ← int8 values
│ Tensor 1: S=0.005, Z=-30     │  ← Quantization params
├──────────────────────────────┤
│ Tensor 2: Dense weights      │  ← int8 values
│ Tensor 2: S=0.031, Z=-64     │  ← Quantization params
├──────────────────────────────┤
│ ...                          │
└──────────────────────────────┘
```

### From .tflite to C Header (.h)
The next step is Phase 2 ─ converting the binary `.tflite` into a C array:

**Command:**
```bash
xxd -i models/magic_wand_model.tflite > model_data.h
```

**Result:**
```c
// The ENTIRE neural network as a list of bytes
const unsigned char models_magic_wand_model_tflite[] = {
  0x1c, 0x00, 0x00, 0x00,  // ← These are the FlatBuffer header bytes
  0x54, 0x46, 0x4c, 0x33,  // ← "TFL3" in ASCII (magic identifier)
  0x00, 0x00, 0x12, 0x00,  // ← Schema/metadata
  // ... thousands of bytes: your weights, biases, S, Z values ...
  0x2f, 0x30, 0x2e, 0x31
};
const unsigned int models_magic_wand_model_tflite_len = 4567;
```

**Why store as C array?**
- Microcontrollers **cannot open files** from a filesystem (no hard drive!)
- The array is stored in **Flash ROM** (persistent memory)
- `const` keyword tells the compiler: "This never changes; put it in ROM, not RAM"
- In C, arrays don't know their own length, so we store `_len` separately

---

## 9. The tf_keras Problem — Why It Exists

### The Timeline
```
TF 2.0 – 2.15:  Keras lived INSIDE TensorFlow
                 → "from tensorflow import keras" ✅
                 → "from tensorflow.keras import layers" ✅
                 → tfmot worked perfectly ✅

TF 2.16+:       Keras became STANDALONE (Keras 3)
                 → Keras 3 supports PyTorch, JAX, AND TensorFlow
                 → But tfmot was NOT updated for Keras 3!
                 → tfmot checks: "Is this a tensorflow.keras.Sequential?"
                 → Keras 3 Sequential ≠ tensorflow.keras.Sequential
                 → ValueError! ❌

Solution:        pip install tf_keras
                 → This is the OLD Keras 2, extracted as a separate package
                 → import tf_keras as keras
                 → Now tfmot recognizes the model type ✅
```

### The Error You Hit
```
ValueError: `prune_low_magnitude` can only prune an object of the following types:
keras.models.Sequential, keras functional model, keras.layers.Layer,
list of keras.layers.Layer.
You passed an object of type: Sequential.
```

It's NOT saying "I don't accept Sequential." It's saying "I expected **tf.keras.Sequential** but got **keras.Sequential** (different Python class, even though they do the same thing)."

### The Fix (What You're Using)
```python
# At the top of ALL your files:
import tf_keras as keras
from tf_keras import layers

# NOT:
# import keras
# from keras import layers
```

### Alternative Fix (Environment Variable)
```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Must be BEFORE importing tf

import tensorflow as tf
# Now tf.keras points to the legacy Keras 2
```

---

## 10. Tools & Next Steps

### Recommended Tools
| Tool | Purpose | How to Use |
|------|---------|-----------|
| **Netron** | Visualize `.tflite` model structure | Drag & drop your `.tflite` file at [netron.app](https://netron.app) |
| **xxd** | Convert binary to C header | `xxd -i model.tflite > model.h` |
| **TF Lite Benchmark Tool** | Measure inference speed | CLI tool from TensorFlow |

### The "Golden Rule" of TinyML
> Always check your **Baseline Accuracy** before optimizing.
> If your model is only 70% accurate, don't prune! Fix the model first.
> Pruning and quantization cause small accuracy drops. Start as high as possible.

### Size Comparison (Your Model)
```
Original Keras (.h5):           ~12.51 KB
After Pruning:                  ~12.51 KB  (same size! zeros are still float32)
After Pruning + Quantization:   ~2-4 KB    (4× smaller + compression)
After gzip compression:         ~1-2 KB    (zeros compress beautifully)
```

---

## 11. Self-Test Quiz 🧪

Test yourself before moving to Phase 2:

### Questions

1. **Why can't you feed a single (x, y, z) reading to a gesture recognition model?**

2. **If your phone samples at 100Hz and a gesture takes 0.5 seconds, what window size do you need?**

3. **Your model has 3,202 parameters. After 50% pruning, how many are zero?**

4. **If you set `end_step=1000` but your training only runs for 300 global steps, what happens?**

5. **A weight of 0.42 with S=0.01 and Z=-30 becomes what int8 value?**

6. **Why is the representative dataset needed for quantization?**

7. **What does `strip_pruning()` actually remove from the model?**

8. **Why do we use `const unsigned char` in the C header instead of `float`?**

<details>
<summary><b>Click for Answers</b></summary>

1. A single point has no temporal information. The model can't distinguish between "sitting on a table" and "mid-wave." You need a **window** of 50+ readings to capture the motion pattern.

2. `100Hz × 0.5s = 50 samples`. Window size = 50.

3. `3,202 × 0.5 = 1,601 weights` are exactly zero.

4. You **never reach 50% sparsity**. The model ends at approximately 10% sparsity. You wasted your time — always calculate `end_step = steps_per_epoch × epochs`.

5. `Q = round(0.42/0.01) + (-30) = 42 + (-30) = 12`

6. Weights are static (ranges known from file). **Activations** are dynamic — they change with every input. The representative dataset lets the converter observe the activation ranges to calculate correct S and Z for each layer.

7. It removes the **pruning masks** (1s and 0s), the **pruning schedule metadata**, and the **wrapper layers** (like `prune_low_magnitude_conv1d`). The zeros remain permanently in the weights.

8. The model is a binary blob (not human-readable floats). `unsigned char` = 1 byte = the smallest addressable unit. `const` forces it into Flash ROM (not precious RAM). On a microcontroller, RAM might be only 256 KB but Flash can be 1 MB+.

</details>

---

> **Next Phase**: Phase 2 — Converting this `.tflite` model into a C header and understanding hexadecimal, memory layout, and FlatBuffers.