# 🔧 Phase 2: From Model to C — Converting `.tflite` to Embedded-Ready Code

> **What this document covers:** Everything from a `.tflite` binary to a C header file ready for a microcontroller.  
> **Prerequisites:** Completed Phase 1 (trained, pruned, quantized `.tflite` model).  
> **Outcome:** You will deeply understand hexadecimal, `xxd`, memory layout, FlatBuffers, and how C code "sees" your neural network.

---

## Table of Contents
*(Will be filled as we learn each topic)*

1. [Hexadecimal — The Language of Bytes](#1-hexadecimal)
2. [xxd — From Binary to C Array](#2-xxd)
3. [Memory Layout — Flash vs RAM](#3-memory-layout)
4. [FlatBuffers — How TFLite Stores a Model](#4-flatbuffers)
5. [The C Header — Reading Your Model as Code](#5-the-c-header)
6. [Verification — Is My Conversion Correct?](#6-verification)

---

## 1. Hexadecimal — The Language of Bytes <a name="1-hexadecimal"></a>

### 1.1 Why Do We Need Another Number System?

You already know **two** number systems:

| System | Base | Digits Used | Example |
|--------|------|-------------|---------|
| **Decimal** | 10 | 0–9 | `255` |
| **Binary** | 2 | 0–1 | `11111111` |

Binary is the computer's native language — every signal is either ON (1) or OFF (0). But binary is **painful for humans to read**:

```
Your TFLite model in binary (just ONE byte):
11111111

Your TFLite model in binary (just FOUR bytes):
11111111 00010100 00000000 00000000

Now imagine reading 3,202 bytes like that...  💀
```

**Hexadecimal (base-16)** is the translator between the human world and the computer world.

> **Analogy:** Think of it like language translation.  
> - **Binary** = the computer's mother tongue (hard for you to read)  
> - **Decimal** = YOUR mother tongue (easy for you, but doesn't map cleanly to bytes)  
> - **Hexadecimal** = a **bilingual shorthand** — easy for you to read, AND maps perfectly to binary

### 1.2 The Hex Digits

Hex uses 16 digits. Since we only have 10 number symbols (0–9), we borrow 6 letters:

```
Decimal:  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15
Hex:      0  1  2  3  4  5  6  7  8  9   A   B   C   D   E   F
Binary:  0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
```

**Key insight:** Each hex digit maps to **exactly 4 binary bits**. This is NOT a coincidence — it's the entire reason hex exists!

$$2^4 = 16 \text{ (possible values with 4 bits = exactly 1 hex digit)}$$

### 1.3 Why Hex Maps Perfectly to Bytes

A **byte** = 8 bits. Since one hex digit = 4 bits:

$$1 \text{ byte} = 8 \text{ bits} = 2 \text{ hex digits}$$

```
┌─── 1 byte (8 bits) ───┐
│  1111  │  1111         │
│  ────  │  ────         │
│   F    │   F           │
└────────┴───────────────┘
        = 0xFF
```

This means **any byte** (0 to 255 in decimal) can be written as **exactly 2 hex digits** (0x00 to 0xFF). Clean, compact, consistent.

> **Analogy:** Decimal is like measuring distance in inches — it works but gets messy.  
> Hex is like measuring in feet — it groups things into neat chunks that match how computers think.

### 1.4 The `0x` Prefix

When you see `0xFF`, the `0x` prefix just means **"this is a hex number."** It's like putting `$` before a price — the `$` isn't part of the number, it just tells you the format.

```python
# Python understands hex natively!
print(0xFF)      # Output: 255
print(0x0A)      # Output: 10
print(0x00)      # Output: 0

# You can also convert:
hex(255)         # Output: '0xff'
bin(0xFF)        # Output: '0b11111111'
```

### 1.5 Conversions — Worked Examples

#### 🔄 Binary → Hex (Group by 4)

**Method:** Split binary into groups of 4 bits (from the right), then look up each group.

**Example:** Convert `11010110` to hex.

```
Step 1: Split into groups of 4
  1101  |  0110

Step 2: Look up each group
  1101 = D    (8+4+0+1 = 13 = D)
  0110 = 6    (0+4+2+0 = 6)

Step 3: Combine
  Answer: 0xD6  ✅
```

#### 🔄 Hex → Binary (Expand each digit)

**Method:** Replace each hex digit with its 4-bit binary equivalent.

**Example:** Convert `0x3A` to binary.

```
Step 1: Take each hex digit
  3       A

Step 2: Convert each to 4 bits
  3 = 0011
  A = 1010

Step 3: Combine
  Answer: 00111010  ✅
```

#### 🔄 Hex → Decimal (Positional value)

**Method:** Each digit × 16^(position), counting from right starting at 0.

**Example:** Convert `0xD6` to decimal.

$$\text{0xD6} = D \times 16^1 + 6 \times 16^0$$
$$= 13 \times 16 + 6 \times 1$$
$$= 208 + 6$$
$$= 214$$

#### 🔄 Decimal → Hex (Repeated division by 16)

**Method:** Divide by 16 repeatedly, collect remainders bottom-to-top.

**Example:** Convert `214` to hex.

```
214 ÷ 16 = 13 remainder 6    →  6  (least significant digit)
 13 ÷ 16 =  0 remainder 13   →  D  (most significant digit)

Read remainders bottom → top: D6
Answer: 0xD6  ✅
```

#### 🔄 Decimal → Hex (Another example with your model!)

Your TFLite model is approximately **3,202 bytes**. What's that in hex?

```
3202 ÷ 16 = 200 remainder 2   →  2
 200 ÷ 16 =  12 remainder 8   →  8
  12 ÷ 16 =   0 remainder 12  →  C

Read remainders bottom → top: C82
Answer: 0xC82  ✅
```

So your model size ≈ `0xC82` bytes. When you see that in memory addresses later, you'll know what it means!

### 1.6 Quick Reference Table

| Decimal | Hex | Binary | Why It Matters |
|---------|-----|--------|----------------|
| 0 | 0x00 | 00000000 | Zero byte — padding, empty memory |
| 10 | 0x0A | 00001010 | Newline character `\n` |
| 13 | 0x0D | 00001101 | Carriage return `\r` |
| 127 | 0x7F | 01111111 | Max signed int8 (your quantized weights!) |
| 128 | 0x80 | 10000000 | Min signed int8 as unsigned = -128 |
| 255 | 0xFF | 11111111 | Max value of 1 byte (unsigned) |

### 1.7 Why This Matters for YOUR TinyML Project

When you convert your `.tflite` model to a C header file using `xxd`, it looks like this:

```c
// This is what YOUR model will look like in C!
unsigned char magic_wand_model_tflite[] = {
  0x20, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33,
  //                       'T'   'F'   'L'   '3'   ← TFLite magic number!
  0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  ...
};
```

Each `0x__` is **one byte** of your neural network. Your quantized weights, biases, the model architecture — all stored as hex bytes.

```
┌─────────────────────────────────────────────────┐
│           Your TFLite Model (~3,202 bytes)      │
│                                                 │
│  Python world:  model.tflite  (binary file)     │
│        ↓  xxd tool                              │
│  C world:       model.h  (hex byte array)       │
│        ↓  compiler                              │
│  Hardware:      Flash memory  (raw bytes)       │
└─────────────────────────────────────────────────┘
```

**The hex representation IS your neural network** — every weight, every bias, every layer configuration, packed into bytes.

### 1.8 Verifying with Python

You can inspect your own `.tflite` model's hex right now:

```python
# Read first 16 bytes of your TFLite model
with open("Phase_1/models/magic_wand_model.tflite", "rb") as f:
    data = f.read(16)

# Show as hex
print("Hex:", data.hex())
print("Bytes:", [f"0x{b:02X}" for b in data])
print("ASCII:", [chr(b) if 32 <= b < 127 else '.' for b in data])
```

The first 4 meaningful bytes should spell **"TFL3"** — that's the TFLite magic number that tells the interpreter "I am a valid TFLite model."

### 1.9 Common Hex Patterns You'll See

| Pattern | Meaning |
|---------|---------|
| `0x00` | Zero — padding or empty |
| `0xFF` | All bits set — could mean -1 (signed) or 255 (unsigned) |
| `0x54 0x46 0x4C 0x33` | ASCII "TFL3" — TFLite file signature |
| `0x7F` | +127 — maximum positive int8 value |
| `0x80` | -128 — minimum int8 value (in two's complement) |

### 1.10 Hex and Your Quantized Weights

Remember from Phase 1: after int8 quantization, every weight in your model is a number between **-128 and +127**. In hex:

$$-128_{10} = \text{0x80} \quad \text{(stored in memory)}$$
$$+127_{10} = \text{0x7F} \quad \text{(stored in memory)}$$
$$0_{10} = \text{0x00} \quad \text{(stored in memory)}$$

So when you look at the hex dump of your model, those `0x__` values are literally your neural network's brain — the quantized weights that decide "Wave" vs "Idle"!

---

### ✅ Quick Check — Test Your Understanding!

**Q1:** Convert `0xB4` to decimal.  
<details>
<summary>Click for answer</summary>

$B \times 16 + 4 = 11 \times 16 + 4 = 176 + 4 = 180$

</details>

**Q2:** How many hex digits do you need to represent 2 bytes?  
<details>
<summary>Click for answer</summary>

4 hex digits. (1 byte = 2 hex digits, so 2 bytes = 4 hex digits)

</details>

**Q3:** You see `0x54 0x46 0x4C 0x33` at the start of a file. What is it?  
<details>
<summary>Click for answer</summary>

It spells "TFL3" in ASCII — the magic number identifying a TFLite model file!  
(0x54='T', 0x46='F', 0x4C='L', 0x33='3')

</details>

**Q4:** A quantized weight in your model is stored as `0xE2`. Is this a positive or negative weight? What's the decimal value?  
<details>
<summary>Click for answer</summary>

`0xE2` = 226 in unsigned. But since int8 is **signed**, values ≥ 128 (0x80) are negative.  
$226 - 256 = -30$  
So it's a **negative weight** with value **-30**.

</details>

---

> 📝 *This document will be filled incrementally as you explore each topic with the Deep Explainer agent.*
