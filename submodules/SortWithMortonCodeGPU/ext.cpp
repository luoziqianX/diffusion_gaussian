#include"./include/ComputeMortonCode.h"
#include"./include/SortByMortonCode.h"
#include<torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("bitonic_sort", &bitonic_sort, "Bitonic sort");
m.def("morton_code", &compute_morton_code, "CUDA sort");
}