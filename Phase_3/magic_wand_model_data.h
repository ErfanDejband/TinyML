#ifndef MAGIC_WAND_MODEL_DATA_H       // ← "If not already included..."
#define MAGIC_WAND_MODEL_DATA_H       // ← "...mark it as included now"

#include <stdalign.h>

// extern alignas(16) const unsigned char magic_wand_model_data[];   // this is for C
alignas(16) extern const unsigned char magic_wand_model_data[]; // now its the C++ format
extern const unsigned int magic_wand_model_data_len;              // ← promise: length exists

#endif                                 // ← "End of include guard"