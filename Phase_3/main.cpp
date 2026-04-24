#include <cstdio>
#include <cstdint>
#include "../phase_2/magic_wand_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// DONE: Step 1 — Allocate tensor arena (Section 3)
constexpr int kTensorArenaSize = 1024 * 10; // 10 KB
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
    // Load model
    const tflite::Model* model = tflite::GetModel(magic_wand_model_data);

    // Create resolver with exactly 6 ops
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddExpandDims();

    // Build interpreter
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena));

    // Allocate tensors
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors() failed!\n");
        return;
    }

    printf("Model initialized successfully!\n");
}

int main() {
    printf("Loading and testing model data...\n");

    const uint8_t* model_data = magic_wand_model_data;
    size_t model_size = magic_wand_model_data_len;

    printf("Model loaded: %zu bytes\n", model_size);
    printf("First 10 bytes of model data: ");
    for (size_t i = 0; i < 10 && i < model_size; ++i) {
        printf("%02x ", model_data[i]);
    }
    printf("\n");

    printf("Starting TFLite Micro inference...\n");
    printf("Tensor arena allocated: %d bytes\n", kTensorArenaSize);

    setup();

    // TODO: Step 4 — Prepare input data (Section 6)
    // TODO: Step 5 — Run inference (Section 7)
    // TODO: Step 6 — Read output (Section 8)

    printf("Inference complete!\n");
    return 0;
}