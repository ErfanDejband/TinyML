#include <cstdio>
#include <cstdint>
#include "../phase_2/magic_wand_model_data.h"

// DONE: Step 1 — Allocate tensor arena (Section 3)
// should define the tensor arena statically, as a global variable, so it persists for the lifetime of the program
// With alignment (optional but recommended):
constexpr int kTensorArenaSize = 1024 * 10; // 10 KB for example
uint8_t tensor_arena[kTensorArenaSize];

int main() {
    printf("Loading and testing model data...\n");

    const uint8_t* model_data = magic_wand_model_data;
    size_t model_size = magic_wand_model_data_len;

    printf("Model loaded: %zu bytes\n", model_size);
    printf("First 10 bytes of model data: ");
    for (size_t i=0; i < 10 && i < model_size; ++i) {
        printf("%02x ", model_data[i]);
    }
    printf("\n");

    printf("Starting TFLite Micro inference...\n");
    printf("Tensor arena allocated: %d bytes\n", kTensorArenaSize);
    
    // TODO: Step 2 — Create op resolver (Section 4)
    
    // TODO: Step 3 — Build interpreter (Section 5)
    
    // TODO: Step 4 — Prepare input data (Section 6)
    
    // TODO: Step 5 — Run inference (Section 7)
    
    // TODO: Step 6 — Read output (Section 8)
    
    printf("Inference complete!\n");
    return 0;
}