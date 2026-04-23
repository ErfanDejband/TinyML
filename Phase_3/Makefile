# ──────────────────────────────────────────────────────────────────────────────
# Phase_3/Makefile — TFLite Micro Inference (using official TFLM build system)
# ──────────────────────────────────────────────────────────────────────────────
#
# This Makefile uses the OFFICIAL TFLite Micro Makefile to build a static
# library, then compiles and links our inference program against it.
#
# Usage (from MSYS2 UCRT64 shell):
#   make          — build everything (TFLM lib + main)
#   make run      — build and run
#   make clean    — remove Phase_3 build artifacts only
#   make clean-all — also clean TFLM library
#
# Prerequisites:
#   - MSYS2 UCRT64 with: gcc, g++, make, git, curl, unzip
#   - tflite-micro cloned at ../tflite-micro
# ──────────────────────────────────────────────────────────────────────────────

# ─── Paths ────────────────────────────────────────────────────────────────────
TFLM_ROOT     := ../tflite-micro
TFLM_MAKE_DIR := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make
TFLM_MAKEFILE := $(TFLM_MAKE_DIR)/Makefile
DOWNLOADS_DIR := $(TFLM_MAKE_DIR)/downloads

# The official Makefile generates output in:
#   gen/{TARGET}_{TARGET_ARCH}_{BUILD_TYPE}_{TOOLCHAIN}/
# On MSYS2 UCRT64: gen/windows_x86_64_default_gcc/
TFLM_LIB_DIR  := $(TFLM_MAKE_DIR)/gen/windows_x86_64_default_gcc/lib
TFLM_LIB      := $(TFLM_LIB_DIR)/libtensorflow-microlite.a

# ─── Compiler ─────────────────────────────────────────────────────────────────
CXX      := g++
CC       := gcc
CXXFLAGS := -std=c++17 -DTF_LITE_STATIC_MEMORY -DTF_LITE_USE_CTIME
CFLAGS   := -std=c17
LDFLAGS  := -lm

# ─── Include paths (same ones the official Makefile uses) ─────────────────────
INCLUDES := \
  -I$(TFLM_ROOT) \
  -I$(DOWNLOADS_DIR) \
  -I$(DOWNLOADS_DIR)/flatbuffers/include \
  -I$(DOWNLOADS_DIR)/gemmlowp \
  -I$(DOWNLOADS_DIR)/ruy

# ─── Our source files ────────────────────────────────────────────────────────
SRCS_CXX := main.cpp
SRCS_C   := ../Phase_2/magic_wand_model_data.c
OBJS     := main.o magic_wand_model_data.o
TARGET   := main

# ─── Rules ────────────────────────────────────────────────────────────────────
.PHONY: all run clean clean-all tflm-lib

all: tflm-lib $(TARGET)

# Build the TFLM static library using the official Makefile
# This auto-downloads flatbuffers, gemmlowp, ruy, etc.
tflm-lib:
	@echo "=== Building TFLite Micro static library (official Makefile) ==="
	$(MAKE) -C $(TFLM_ROOT) -f tensorflow/lite/micro/tools/make/Makefile microlite
	@echo "=== TFLM library ready ==="

# Link our program against the TFLM static library
$(TARGET): $(OBJS) $(TFLM_LIB)
	$(CXX) $(CXXFLAGS) $(OBJS) $(TFLM_LIB) -o $(TARGET) $(LDFLAGS)
	@echo "=== Build complete: ./$(TARGET) ==="

# Compile our C++ source
main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile the model data (plain C)
magic_wand_model_data.o: ../Phase_2/magic_wand_model_data.c
	$(CC) $(CFLAGS) -c $< -o $@

# Build and run
run: all
	./$(TARGET)

# Clean only Phase_3 artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Clean everything including TFLM library
clean-all: clean
	$(MAKE) -C $(TFLM_ROOT) -f tensorflow/lite/micro/tools/make/Makefile clean
