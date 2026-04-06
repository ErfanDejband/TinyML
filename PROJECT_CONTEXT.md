# 🧠 TinyML Project Context (Shared Brain File)

> **Purpose:** Any Copilot Chat session can reference this file with `#file:PROJECT_CONTEXT.md`
> to instantly have full context of this project.

## What I'm Building
- **Smartphone gesture recognizer** (Wave vs Idle)
- **Pipeline:** Phone accelerometer → JSON → CSV → Sliding Windows → CNN → Pruning → Quantization → TFLite → C Header → TFLite Micro (C++)

## My Skill Level
- ✅ **Python:** Proficient (Pandas, NumPy, TensorFlow/Keras, sklearn)
- 🟡 **C/C++:** Basic (learning through this project)
- 🟡 **Embedded Systems:** Beginner (learning TinyML concepts)

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | TinyML Foundations (Data, Windowing, CNN, Pruning, Quantization) | ✅ Complete |
| **Phase 2** | From Python to C (xxd export, .h header, FlatBuffer understanding) | ✅ Complete (Ready for Phase 3) |
| **Phase 3** | From Basic C to Embedded C++ (TFLite Micro, Tensor Arena, Inference) | 🔄 Next |
| **Phase 4** | Hardware Implementation (Real-time on phone/Arduino) | ⏳ Future |

## Project Files

| File | Purpose |
|------|---------|
| `process_data.py` | Reads JSON sensor data from phone, creates CSV with x,y,z,label |
| `prepare_data_for_training.py` | Creates sliding windows (50 timesteps, 50% overlap), encodes labels, stratified split |
| `create_model.py` | Builds CNN: Conv1D(8) → MaxPool → Flatten → Dense(16) → Dense(2, softmax) |
| `optimize_model.py` | Applies pruning (50% ConstantSparsity) + full int8 quantization → .tflite |
| `Phase_1_TinyML_Foundations.md` | Comprehensive Phase 1 learning material (math, diagrams, theory) |
| `PROJECT_CONTEXT.md` | This file — shared context for all chat sessions |

## Folder Structure
```
TinyML/
├── PROJECT_CONTEXT.md              ← You are here
├── .github/
│   └── prompts/
│       ├── code-helper.prompt.md
│       ├── deep-explainer.prompt.md
│       └── roadmap-manager.prompt.md
├── Phase_1/                        ← ✅ COMPLETE
│   ├── Phase_1_TinyML_Foundations.md
│   ├── TinyML.ipynb
│   ├── process_data.py
│   ├── prepare_data_for_training.py
│   ├── create_model.py
│   ├── optimize_model.py
│   ├── RowData/
│   └── models/
│       ├── my_model.h5
│       └── magic_wand_model.tflite
├── Phase_2/                        ← 🔄 IN PROGRESS
│   └── Phase_2_From_Model_to_C.md
├── Phase_3/                        ← 🔜 Future
└── Phase_4/                        ← 🔜 Future
```

## Current Status
| Phase | Description | Status |
|-------|------------|--------|
| Phase 1 | TinyML Foundations (Data, CNN, Pruning, Quantization) | ✅ Complete |
| Phase 2 | From Model to C (xxd, hex, FlatBuffers, `.h` header) | ✅ Complete — All 6 sections learned, C headers created |
| Phase 3 | Embedded C/C++ (TFLite Micro, Tensor Arena) | 🔄 Ready to start |
| Phase 4 | Hardware Implementation (real-time on device) | 🔜 Not started |

## My Model Specs
- **Input shape:** (50, 3) — 50 timesteps, 3 axes (x, y, z)
- **Architecture:** Conv1D(8, kernel=3) → MaxPooling1D(2) → Flatten → Dense(16) → Dense(2, softmax)
- **Total parameters:** 3,202 (12.51 KB)
- **Training data:** 478 train samples, 120 test samples
- **Steps per epoch:** 15 (478 samples ÷ 32 batch size, ceiling)
- **Training result:** 100% validation accuracy by epoch 8 (possible overfitting — small dataset)
- **After quantization:** ~2-4 KB (.tflite), full int8

## Key Technical Decisions
1. Using `tf_keras` (not Keras 3) because `tensorflow_model_optimization` requires legacy Keras
2. Full int8 quantization (not float16) — microcontrollers have no FPU
3. 50% pruning with `ConstantSparsity` schedule
4. Representative dataset: 100 samples for activation range calibration
5. `converter.inference_input_type = tf.int8` and `converter.inference_output_type = tf.int8`

## Known Issues
- ⚠️ Possible overfitting (100% accuracy with only 478 samples — need more data)
- ⚠️ Mutable default arguments in `create_model.py` (`List[int] = [16]`)
- ✅ Fixed: `tf_keras` import issue (was `ModuleNotFoundError`)

## Key Concepts I've Mastered

### Phase 1: TinyML Foundations
- Sliding windows and why single data points are meaningless for gesture recognition
- CNN architecture for time-series (Conv1D, not Conv2D)
- Pruning: masks, schedules (ConstantSparsity vs PolynomialDecay), callbacks, stripping
- Quantization: Scale (S), Zero Point (Z), per-layer calibration, representative dataset
- Steps vs Epochs vs Batches (478 samples ÷ 32 batch = 15 steps/epoch)
- Why pruning alone doesn't reduce file size (needs compression)
- Why quantization needs representative data (for activation ranges, not weight ranges)
- TFLite FlatBuffer format vs Keras .h5 format

### Phase 2: From Model to C
- Hexadecimal number system and why it maps perfectly to bytes (4 bits = 1 hex digit)
- The xxd tool for converting binary files to C arrays
- Flash ROM vs RAM memory (const keyword, why models live in Flash)
- FlatBuffers: vtables, offsets, inline vs referenced data, root table navigation
- C header files: include guards, extern declarations, alignas for optimization
- Model verification techniques (byte comparison, FlatBuffer structure validation)

## What I'm Learning Next (Phase 3)
- Setting up TFLite Micro library in a C++ project
- Understanding tensor arenas and memory allocation strategies
- Configuring the MicroOpResolver with required operators
- Writing C++ inference code and handling input/output tensors
- Comparing Python vs C++ inference results for verification
- Optimizing tensor arena size and understanding memory constraints