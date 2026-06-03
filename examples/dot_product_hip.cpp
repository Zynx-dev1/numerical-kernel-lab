#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

__global__ void dot_kernel(const float* a, const float* b, float* out, int n) {
    __shared__ float cache[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    float temp = (idx < n) ? a[idx] * b[idx] : 0.0f;
    cache[tid] = temp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) cache[tid] += cache[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, cache[0]);
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f);
    float h_out = 0.0f;
    float *d_a, *d_b, *d_out;
    hipMalloc(&d_a, bytes); hipMalloc(&d_b, bytes); hipMalloc(&d_out, sizeof(float));
    hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_out, &h_out, sizeof(float), hipMemcpyHostToDevice);
    int block = 256, grid = (N + block - 1) / block;
    hipLaunchKernelGGL(dot_kernel, dim3(grid), dim3(block), 0, 0, d_a, d_b, d_out, N);
    hipDeviceSynchronize();
    hipMemcpy(&h_out, d_out, sizeof(float), hipMemcpyDeviceToHost);
    bool ok = (h_out == static_cast<float>(N) * 2.0f);
    std::cout << "[dot_product_hip] N=" << N << " result=" << (ok ? "PASS" : "FAIL") << "\n";
    hipFree(d_a); hipFree(d_b); hipFree(d_out);
    return ok ? 0 : 1;
}
