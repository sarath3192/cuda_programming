#include <iostream>
#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

#define M 16
#define N 16
#define K 16

__global__ void tensorCoreGEMM(const half* a, const half* b, float* c) {
    // Each warp does one matrix multiply
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    // Load and compute
    wmma::load_matrix_sync(a_frag, a, K);
    wmma::load_matrix_sync(b_frag, b, N);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}

int main() {
    half *a_host, *b_host;
    float *c_host;
    half *a_dev, *b_dev;
    float *c_dev;

    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(float);

    // Allocate and init host memory
    a_host = new half[M * K];
    b_host = new half[K * N];
    c_host = new float[M * N];

    for (int i = 0; i < M * K; i++) a_host[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) b_host[i] = __float2half(1.0f);

    // Device memory
    cudaMalloc(&a_dev, sizeA);
    cudaMalloc(&b_dev, sizeB);
    cudaMalloc(&c_dev, sizeC);

    cudaMemcpy(a_dev, a_host, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, sizeB, cudaMemcpyHostToDevice);

    tensorCoreGEMM<<<1, 32>>>(a_dev, b_dev, c_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(c_host, c_dev, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "C[0] = " << c_host[0] << std::endl; // Should be 16.0

    // Cleanup
    delete[] a_host;
    delete[] b_host;
    delete[] c_host;
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);

    return 0;
}
