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
#   - tflite-micro cloned at ../tflite-micro (with LF line endings!)
#     git clone -c core.autocrlf=false https://github.com/tensorflow/tflite-micro.git
# ──────────────────────────────────────────────────────────────────────────────

# ─── Paths ────────────────────────────────────────────────────────────────────
TFLM_ROOT     := ../tflite-micro
TFLM_MAKE_DIR := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make
TFLM_MAKEFILE := $(TFLM_MAKE_DIR)/Makefile
DOWNLOADS_DIR := $(TFLM_MAKE_DIR)/downloads

# The official Makefile detects OS/arch automatically and generates output in:
#   tflite-micro/gen/{TARGET}_{ARCH}_{BUILD_TYPE}_{TOOLCHAIN}/lib/
# We use a wildcard to find it, or you can hardcode after first build.
TFLM_GEN_DIR  := $(wildcard $(TFLM_ROOT)/gen/*/lib)
TFLM_LIB      := $(TFLM_GEN_DIR)/libtensorflow-microlite.a

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
TARGET_BIN := main
SRCS_CXX   := main.cpp
SRCS_C     := ../Phase_2/magic_wand_model_data.c
OBJS       := main.o magic_wand_model_data.o

# ─── Rules ────────────────────────────────────────────────────────────────────
.PHONY: all run clean clean-all tflm-lib

all: tflm-lib $(TARGET_BIN)

# Build the TFLM static library using the official Makefile.
# This auto-downloads flatbuffers, gemmlowp, ruy, etc.
tflm-lib:
	@echo "=== Building TFLite Micro static library (official Makefile) ==="
	$(MAKE) -C $(TFLM_ROOT) -f tensorflow/lite/micro/tools/make/Makefile microlite
	@echo "=== TFLM library ready ==="

# After tflm-lib builds, re-evaluate TFLM_GEN_DIR (it may not exist on first run)
$(TARGET_BIN): $(OBJS)
	$(eval TFLM_GEN_DIR := $(wildcard $(TFLM_ROOT)/gen/*/lib))
	$(eval TFLM_LIB := $(TFLM_GEN_DIR)/libtensorflow-microlite.a)
	$(CXX) $(CXXFLAGS) $(OBJS) $(TFLM_LIB) -o $(TARGET_BIN) $(LDFLAGS)
	@echo "=== Build complete: ./$(TARGET_BIN) ==="

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

magic_wand_model_data.o: ../Phase_2/magic_wand_model_data.c
	$(CC) $(CFLAGS) -c $< -o $@

run: all
	./$(TARGET_BIN)

clean:
	rm -f $(OBJS) $(TARGET_BIN)

clean-all: clean
	$(MAKE) -C $(TFLM_ROOT) -f tensorflow/lite/micro/tools/make/Makefile clean
