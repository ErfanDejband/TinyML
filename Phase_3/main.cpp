// hello.cpp
#include <cstdio>
#include <cstdint>

int main() {
    uint8_t value = 100;
    const uint8_t constarray[6] = {1, 2, 3, 4, 5, 6};
    size_t size_array = sizeof(constarray) / sizeof(constarray[0]);
    printf("Value: %u\n", value);
    printf("Array size: %zu\n", size_array);
    printf("Array elements: ");
    for (size_t i = 0; i < size_array; ++i) {
        printf("%u ", constarray[i]);
    }
    printf("\n");

    return 0;
}