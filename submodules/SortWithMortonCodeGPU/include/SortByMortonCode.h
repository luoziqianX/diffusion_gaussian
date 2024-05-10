//
// Created by Ziqian Luo on 24-5-7.
//

#ifndef SORTWITHMORTONCODEGPU_SORTBYMORTONCODE_H
#define SORTWITHMORTONCODEGPU_SORTBYMORTONCODE_H

#include<torch/extension.h>

void bitonic_sort(
        torch::Tensor &means3d, torch::Tensor &features_dc,
        torch::Tensor &scales, torch::Tensor &rotations,
        torch::Tensor &opacities, torch::Tensor &morton_codes
);

#endif //SORTWITHMORTONCODEGPU_SORTBYMORTONCODE_H
