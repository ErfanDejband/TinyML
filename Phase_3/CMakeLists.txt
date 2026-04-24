# CMakeLists.txt — Your build recipe
cmake_minimum_required(VERSION 3.16)
project(TinyML_Main)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TFLM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-micro)
set(FB_ROOT   ${CMAKE_CURRENT_SOURCE_DIR}/../flatbuffers)
set(TFLM_SOURCES
    # --- Core (the interpreter engine) ---
    ${TFLM_ROOT}/tensorflow/lite/micro/micro_interpreter.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/micro_interpreter_graph.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/micro_interpreter_context.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/micro_allocator.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/micro_allocation_info.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/micro_context.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/micro_op_resolver.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/micro_utils.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/memory_helpers.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/flatbuffer_utils.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/micro_log.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/debug_log.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/system_setup.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/micro_time.cc

    # --- Memory planners ---
    ${TFLM_ROOT}/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cc

    # --- Arena allocators ---
    ${TFLM_ROOT}/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc

    # --- Bridge ---
    ${TFLM_ROOT}/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc

    # --- Op kernels (6 ops: Conv2D, MaxPool2D, Reshape, FullyConnected, Softmax, ExpandDims) ---
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/conv.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/conv_common.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/pooling.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/pooling_common.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/reshape.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/reshape_common.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/fully_connected.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/fully_connected_common.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/softmax.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/softmax_common.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/expand_dims.cc
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/kernel_util.cc

    # --- Link dependencies ---
    ${TFLM_ROOT}/tensorflow/lite/micro/micro_resource_variable.cc
    ${TFLM_ROOT}/tensorflow/compiler/mlir/lite/schema/schema_utils.cc
    ${TFLM_ROOT}/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc
    ${TFLM_ROOT}/tensorflow/lite/array.cc

    # --- Shared TFLite core ---
    ${TFLM_ROOT}/tensorflow/lite/core/api/flatbuffer_conversions.cc
    ${TFLM_ROOT}/tensorflow/lite/core/api/tensor_utils.cc
    ${TFLM_ROOT}/tensorflow/lite/core/c/common.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/kernel_util.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/common.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/quantization_util.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/portable_tensor_utils.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/tensor_utils.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/tensor_ctypes.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/runtime_shape.cc
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc
)
include_directories(
    ${TFLM_ROOT}
    ${FB_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../gemmlowp
    ${CMAKE_CURRENT_SOURCE_DIR}/../ruy
)

# Create executable from main.cpp + all TFLite Micro sources
add_executable(main 
    main.cpp 
    ../Phase_2/magic_wand_model_data.c
    ${TFLM_SOURCES}
)