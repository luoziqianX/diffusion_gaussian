#include<torch/extension.h>

__global__ void
morton_code_cuda_kernel(
        const float *__restrict__ means3d,
        int *__restrict__ morton_codes,
        int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = means3d[i * 3 + 0];
        float y = means3d[i * 3 + 1];
        float z = means3d[i * 3 + 2];
        int xx = (int) (x * 1024);
        int yy = (int) (y * 1024);
        int zz = (int) (z * 1024);

        morton_codes[i] = 0;
        for (int j = 0; j < 10; j++) {
            morton_codes[i] |= (xx & 1) << (3 * j);
            morton_codes[i] |= (yy & 1) << (3 * j + 1);
            morton_codes[i] |= (zz & 1) << (3 * j + 2);
            xx >>= 1;
            yy >>= 1;
            zz >>= 1;
        }
    }
}

void compute_morton_code(
        torch::Tensor &means3d,
        torch::Tensor &morton_codes
) {
    int N = means3d.size(0);
    morton_code_cuda_kernel<<<(N + 1023) / 1024, 1024>>>(
            means3d.data_ptr<float>(),
            morton_codes.data_ptr<int>(),
            N
    );
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("error in morton_code_cuda_kernel: %s\n", cudaGetErrorString(err));
            exit(1);
        }
    }
}