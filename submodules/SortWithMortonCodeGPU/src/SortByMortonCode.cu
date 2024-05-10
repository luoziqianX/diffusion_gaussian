#include<torch/extension.h>

__device__ void swap(
        float *means3d, float *features_dc, float *scales,
        float *rotations, float *opacities,
        int *morton_codes, int i, int j
) {
    float tmp;
    tmp = means3d[i * 3];
    means3d[i * 3] = means3d[j * 3];
    means3d[j * 3] = tmp;

    tmp = means3d[i * 3 + 1];
    means3d[i * 3 + 1] = means3d[j * 3 + 1];
    means3d[j * 3 + 1] = tmp;

    tmp = means3d[i * 3 + 2];
    means3d[i * 3 + 2] = means3d[j * 3 + 2];
    means3d[j * 3 + 2] = tmp;

    tmp = features_dc[i * 3];
    features_dc[i * 3] = features_dc[j * 3];
    features_dc[j * 3] = tmp;

    tmp = features_dc[i * 3 + 1];
    features_dc[i * 3 + 1] = features_dc[j * 3 + 1];
    features_dc[j * 3 + 1] = tmp;

    tmp = features_dc[i * 3 + 2];
    features_dc[i * 3 + 2] = features_dc[j * 3 + 2];
    features_dc[j * 3 + 2] = tmp;

    tmp = scales[i * 3];
    scales[i * 3] = scales[j * 3];
    scales[j * 3] = tmp;

    tmp = scales[i * 3 + 1];
    scales[i * 3 + 1] = scales[j * 3 + 1];
    scales[j * 3 + 1] = tmp;

    tmp = scales[i * 3 + 2];
    scales[i * 3 + 2] = scales[j * 3 + 2];
    scales[j * 3 + 2] = tmp;

    for (int l = 0; l < 4; l++) {
        tmp = rotations[i * 4 + l];
        rotations[i * 4 + l] = rotations[j * 4 + l];
        rotations[j * 4 + l] = tmp;
    }

    tmp = opacities[i];
    opacities[i] = opacities[j];
    opacities[j] = tmp;

    int tmp_int = morton_codes[i];
    morton_codes[i] = morton_codes[j];
    morton_codes[j] = tmp_int;
}

__global__ void
bitonic_sort_cuda_kernel(
        float *means3d, float *features_dc,
        float *scales, float *rotations,
        float *opacities, int *morton_codes,
        int stride, int inner_stride
) {
    unsigned int flipper = inner_stride >> 1;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int tid_other = tid ^ flipper;
    if (tid >= tid_other) {
        return;
    }
    if ((tid & stride) == 0) {
        if (morton_codes[tid] > morton_codes[tid_other]) {
            swap(means3d, features_dc, scales, rotations, opacities, morton_codes, tid, tid_other);
        }
    } else {
        if (morton_codes[tid] < morton_codes[tid_other]) {
            swap(means3d, features_dc, scales, rotations, opacities, morton_codes, tid, tid_other);
        }
    }
}

void bitonic_sort(
        torch::Tensor &means3d, torch::Tensor &features_dc,
        torch::Tensor &scales, torch::Tensor &rotations,
        torch::Tensor &opacities, torch::Tensor &morton_codes
) {
    int length = means3d.size(0);
    auto means3d_ptr = means3d.data_ptr<float>();
    auto features_dc_ptr = features_dc.data_ptr<float>();
    auto scales_ptr = scales.data_ptr<float>();
    auto rotations_ptr = rotations.data_ptr<float>();
    auto opacities_ptr = opacities.data_ptr<float>();
    auto morton_codes_ptr = morton_codes.data_ptr<int>();
    unsigned int twoUpper = 1;
    for (; twoUpper < length; twoUpper <<= 1) {
        if (twoUpper == length) {
            break;
        }
    }

    float *input_means3d_device;
    float *input_features_dc_device;
    float *input_scales_device;
    float *input_rotations_device;
    float *input_opacities_device;
    int *input_morton_codes_device;
    unsigned int input_length;
    if (length == twoUpper) {
        input_length = twoUpper;
        input_means3d_device = means3d_ptr;
        input_features_dc_device = features_dc_ptr;
        input_scales_device = scales_ptr;
        input_rotations_device = rotations_ptr;
        input_opacities_device = opacities_ptr;
        input_morton_codes_device = morton_codes_ptr;
    } else {
        input_length = twoUpper;
        cudaMalloc(&input_means3d_device, twoUpper * 3 * sizeof(float));
        cudaMalloc(&input_features_dc_device, twoUpper * 3 * sizeof(float));
        cudaMalloc(&input_scales_device, twoUpper * 3 * sizeof(float));
        cudaMalloc(&input_rotations_device, twoUpper * 4 * sizeof(float));
        cudaMalloc(&input_opacities_device, twoUpper * sizeof(float));
        cudaMalloc(&input_morton_codes_device, twoUpper * sizeof(int));

        cudaMemcpy(input_means3d_device, means3d_ptr, length * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(input_features_dc_device, features_dc_ptr, length * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(input_scales_device, scales_ptr, length * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(input_rotations_device, rotations_ptr, length * 4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(input_opacities_device, opacities_ptr, length * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(input_morton_codes_device, morton_codes_ptr, length * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemset(input_means3d_device + length * 3, 0, (twoUpper - length) * 3 * sizeof(float));
        cudaMemset(input_features_dc_device + length * 3, 0, (twoUpper - length) * 3 * sizeof(float));
        cudaMemset(input_scales_device + length * 3, 0, (twoUpper - length) * 3 * sizeof(float));
        cudaMemset(input_rotations_device + length * 4, 0, (twoUpper - length) * 4 * sizeof(float));
        cudaMemset(input_opacities_device + length, 0, (twoUpper - length) * sizeof(float));
        cudaMemset(input_morton_codes_device + length, -1, (twoUpper - length) * sizeof(int));
    }

    dim3 grid_dim((input_length / 256 == 0) ? 1 : input_length / 256, 1, 1);
    dim3 block_dim((input_length / 256 == 0) ? input_length : 256, 1, 1);
    for (unsigned int stride = 2; stride <= input_length; stride <<= 1) {
        for (unsigned int inner_stride = stride; inner_stride >= 2; inner_stride >>= 1) {
            bitonic_sort_cuda_kernel<<<grid_dim, block_dim>>>(input_means3d_device, input_features_dc_device,
                                                         input_scales_device, input_rotations_device,
                                                         input_opacities_device, input_morton_codes_device,
                                                         stride, inner_stride);
        }
    }


    if (twoUpper == length) {
        return;
    } else {
        cudaMemcpy(means3d_ptr, input_means3d_device + (twoUpper - length) * 3, length * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(features_dc_ptr, input_features_dc_device + (twoUpper - length) * 3, length * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(scales_ptr, input_scales_device + (twoUpper - length) * 3, length * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(rotations_ptr, input_rotations_device + (twoUpper - length) * 4, length * 4 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(opacities_ptr, input_opacities_device + (twoUpper - length), length * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(morton_codes_ptr, input_morton_codes_device + (twoUpper - length), length * sizeof(int),
                   cudaMemcpyDeviceToHost);

        cudaFree(input_means3d_device);
        cudaFree(input_features_dc_device);
        cudaFree(input_scales_device);
        cudaFree(input_rotations_device);
        cudaFree(input_opacities_device);
        cudaFree(input_morton_codes_device);
    }
}