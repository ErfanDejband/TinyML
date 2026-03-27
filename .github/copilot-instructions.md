# TinyML Project — Copilot Instructions

## Project Overview
This is a TinyML gesture recognition project (Wave vs Idle) using smartphone accelerometer data.
The student is learning the full pipeline: Python training → Pruning → Quantization → TFLite → C → Hardware.

## IMPORTANT: Always read these files for context
- `PROJECT_CONTEXT.md` — Current status, model specs, decisions, known issues
- Based on the phase you are working on, read the related markdown file for that phase. For example, if you are working on Phase 2, read the markdown file in the Phase_2 folder that starts with Phase_2_. This file will has give you the idea what we learned.
- All other .md files in the root directory

## Student Profile
- **Python**: Good / comfortable
- **C/C++**: Basic — actively learning
- **Goal**: Master TinyML end-to-end for career growth
- **Learning style**: Needs intuitive analogies, worked math examples, ASCII diagrams
- **Pet peeves**: Don't rush to the next topic. Don't give long answers when a short one works.

## Code Style
- Use type hints in Python
- Use `tf_keras` (not `keras`) for all TensorFlow code (tfmot compatibility)
- Follow existing project patterns (see `create_model.py` for reference)
- Always explain WHY, not just HOW