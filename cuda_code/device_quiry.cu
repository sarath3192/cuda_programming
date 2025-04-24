#include <iostream>
#include <cuda_runtime.h>

void checkCudaDevice() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "CUDA device not found!" << std::endl;
        return;
    }

    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    std::cout << "Using CUDA device: " << deviceProp.name << std::endl;
    std::cout << "Total Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
}

int main() {
    checkCudaDevice();
    return 0;
}
