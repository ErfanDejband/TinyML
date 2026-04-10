#include <cstdio>
#include <cstdint>
#include "../phase_2/magic_wand_model_data.h"

int main() {
    // Now you can use the model!
    const uint8_t* model_data = magic_wand_model_data;
    size_t model_size = magic_wand_model_data_len;
    
    printf("Model loaded: %zu bytes\n", model_size);
    for (size_t i=0; i < 5 && i < model_size; ++i) {
        printf("%02x ", model_data[i]);
    }
    printf("\n");

}