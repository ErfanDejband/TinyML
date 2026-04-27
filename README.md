# TinyML — Gesture Recognition (Wave vs Idle)

A TinyML project learning the full pipeline: Phone Sensor → Python Training → Pruning → Quantization → TFLite → C → Hardware.

## 🚀 Getting Started

### Option 1: Poetry (Recommended)

Poetry provides better dependency management, reproducible environments, and prevents version conflicts.

```powershell
# Set Python interpreter (Windows)
poetry env use C:\Users\YourName\AppData\Local\Programs\Python\Python312\python.exe

# Install dependencies
poetry install

# Activate environment
poetry shell

# Run scripts
python Phase_1/create_model.py
```

**Why Poetry?**
- ✅ Automatic dependency resolution (no conflicts)
- ✅ Lock file ensures everyone gets identical versions
- ✅ Modern tooling (`poetry add`, `poetry update`)
- ✅ Separates dev vs production dependencies

### Option 2: pip + requirements.txt

Traditional approach, works fine for simple projects.

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run scripts
python Phase_1/create_model.py
```

**Note:** This setup uses slightly newer package versions that may have undeclared dependency conflicts. Works in practice but less reliable than Poetry.

## ⚡ Phase 3: C++ Inference (TFLite Micro)

Phase 3 runs the trained model in C++ using TFLite Micro. This requires **MSYS2 UCRT64** on Windows.

### Prerequisites

1. Install [MSYS2](https://www.msys2.org/) and open the **UCRT64** terminal
2. Install build tools:
   ```bash
   pacman -S --noconfirm \
     mingw-w64-ucrt-x86_64-gcc \
     mingw-w64-ucrt-x86_64-python \
     mingw-w64-ucrt-x86_64-python-numpy \
     mingw-w64-ucrt-x86_64-python-pillow \
     make git curl unzip patch diffutils
   ```
3. Clone tflite-micro (must use LF line endings):
   ```bash
   git clone -c core.autocrlf=false https://github.com/tensorflow/tflite-micro.git
   ```

### Build & Run

```bash
cd Phase_3
make run
```

First build takes 5–15 min. See `Phase_3/Phase_3_TFLite_Micro_Inference.md` Section 4.5.5 for detailed setup guide.

## 🤖 Copilot Prompt Agents

This project includes 3 specialized Copilot prompt agents in `.github/prompts/`. Open Copilot Chat and type `/` to use them:

| Command | Role | Ask it... |
|---------|------|-----------|
| `/roadmap-manager` | Learning supervisor | "What's my next step?" |
| `/code-helper` | Code writer & debugger | "Fix this error..." |
| `/deep-explainer` | Math & theory teacher | "Explain quantization math" |

All agents read `PROJECT_CONTEXT.md` automatically for shared context.
