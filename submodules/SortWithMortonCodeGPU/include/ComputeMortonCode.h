//
// Created by Ziqian Luo on 24-5-7.
//

#ifndef SORTWITHMORTONCODEGPU_COMPUTEMORTONCODE_H
#define SORTWITHMORTONCODEGPU_COMPUTEMORTONCODE_H

#include<torch/extension.h>

void compute_morton_code(
        torch::Tensor &means3d,
        torch::Tensor &morton_codes
);

#endif //SORTWITHMORTONCODEGPU_COMPUTEMORTONCODE_H
