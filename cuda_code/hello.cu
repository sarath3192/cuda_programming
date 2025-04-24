#include <stdio.h> // This a standard library

__global__ void hello_cuda() { //
// In CUDA programming, __global__ is a function qualifier used to declare a kernel function.
//  A kernel function is a function that runs on the GPU and is called from the host (CPU) code.
// Key Points:
// Kernel Launch Syntax: kernel_name<<<gridDim, blockDim>>>(args);

// gridDim = number of blocks

// blockDim = number of threads per block

// __global__ can only be used with void return type (no return values).

// Memory transfer is required between host and device before/after calling these functions.

    printf("Hello from the GPU!\n");
}

int main() {
    hello_cuda<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
