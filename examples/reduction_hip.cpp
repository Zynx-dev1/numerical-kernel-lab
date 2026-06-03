#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

__global__ void reduction_sum(const float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    std::vector<float> h_in(N, 1.0f);
    float h_out = 0.0f;
    float *d_in, *d_out;
    hipMalloc(&d_in, bytes); hipMalloc(&d_out, sizeof(float));
    hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_out, &h_out, sizeof(float), hipMemcpyHostToDevice);
    int block = 256, grid = (N + block - 1) / block;
    hipLaunchKernelGGL(reduction_sum, dim3(grid), dim3(block), 0, 0, d_in, d_out, N);
    hipDeviceSynchronize();
    hipMemcpy(&h_out, d_out, sizeof(float), hipMemcpyDeviceToHost);
    bool ok = (h_out == static_cast<float>(N));
    std::cout << "[reduction_hip] N=" << N << " sum=" << h_out << " result=" << (ok ? "PASS" : "FAIL") << "\n";
    hipFree(d_in); hipFree(d_out);
    return ok ? 0 : 1;
}
