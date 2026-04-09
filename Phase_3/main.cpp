#include <cstdio>
#include <cstdint>
#include "./magic_wand_model_data.h"

int main() {
    printf("Model size: %u bytes\n", magic_wand_model_data_len);
    printf("First 4 bytes: ");
    for (int i = 0; i < 4; i++) {
        printf("0x%02X ", magic_wand_model_data[i]);
    }
    printf("\n");
    
    return 0;
}