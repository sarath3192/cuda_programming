# APIs at Each Layer

    CUDA Runtime API (High-level)
    Used by developers in user space. Abstracts the GPU driver internals.

**Common APIs:**

    1. cudaMemcpy(), cudaMalloc(), cudaFree()

    2. cudaLaunchKernel() / <<<>>> kernel syntax

    3. cudaDeviceSynchronize()

    4. cudaStreamCreate(), cudaStreamDestroy()


# 2. CUDA Driver API (Low-level):

    More control and flexibility than the runtime API, closer to how drivers actually work.

**Core APIs:**

    1. cuInit(), cuDeviceGet(), cuCtxCreate()

    2. cuMemAlloc(), cuMemFree(), cuMemcpyHtoD()

    3. cuModuleLoad(), cuModuleGetFunction()

    4. cuLaunchKernel()


# 3. Features:

| Feature   | Runtime API  | Driver API
| --------  | -----------  | ----------
| Simpler, less boilerplate   | ✅            | ❌
| Fine-grained control  | ❌           | ✅
| Kernel launch syntax    |    kernel<<<>>>           | cuLaunchKernel()
| Manual memory management    |    Partial           | Required

# 3. CUDA driver api code
  
        // File: main_driver.cpp
        #include <cuda.h>
        #include <iostream>

        #define checkCudaError(val) check((val), #val, __FILE__, __LINE__)
        void check(CUresult err, const char* fn, const char* file, int line) {
            if (err != CUDA_SUCCESS) {
                const char* errStr;
                cuGetErrorString(err, &errStr);
                std::cerr << "CUDA Error: " << errStr << " at " << file << ":" << line << " in " << fn << std::endl;
                exit(1);
            }
        }

        int main() {
        const int N = 10;
        int h_data[N] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

        // Initialize
        checkCudaError(cuInit(0));

        CUdevice device;
        checkCudaError(cuDeviceGet(&device, 0));

        CUcontext context;
        checkCudaError(cuCtxCreate(&context, 0, device));

        // Load PTX
        CUmodule module;
        checkCudaError(cuModuleLoad(&module, "add_one.ptx"));

        CUfunction kernel;
        checkCudaError(cuModuleGetFunction(&kernel, module, "addOne"));

        // Allocate device memory
        CUdeviceptr d_data;
        checkCudaError(cuMemAlloc(&d_data, N * sizeof(int)));
        checkCudaError(cuMemcpyHtoD(d_data, h_data, N * sizeof(int)));

        // Setup kernel args
        void* args[] = { &d_data, (void*)&N };

        // Launch kernel
        checkCudaError(cuLaunchKernel(kernel,
                                    1, 1, 1,     // gridDim
                                    N, 1, 1,     // blockDim
                                    0, 0,        // sharedMem, stream
                                    args, 0));   // args

        checkCudaError(cuCtxSynchronize());

        checkCudaError(cuMemcpyDtoH(h_data, d_data, N * sizeof(int)));

        for (int i = 0; i < N; ++i)
            std::cout << h_data[i] << " ";
        std::cout << std::endl;

        cuMemFree(d_data);
        cuCtxDestroy(context);

        return 0;} 

# 4. CUDA Runtime api example

        // File: main_runtime.cpp
        #include <iostream>
        #include <cuda_runtime.h>

        __global__ void addOne(int* data, int n) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n)
                data[idx] += 1;
        }

        int main() {
        const int N = 10;
        int h_data[N] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        int *d_data;

        cudaMalloc((void**)&d_data, N * sizeof(int));
        cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

        addOne<<<1, N>>>(d_data, N);
        cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);

        for (int i = 0; i < N; ++i)
            std::cout << h_data[i] << " ";
        std::cout << std::endl;

        return 0;}


# 5. What is the Vulkan “runtime”?

While we often call it a "runtime," what Vulkan actually has is a loader and driver infrastructure — not a separate runtime environment like Java or Python.

***This tutorial will teach you the basics of using the Vulkan graphics and compute API. Vulkan is a new API by the Khronos group (known for OpenGL) that provides a much better abstraction of modern graphics cards. This new interface allows you to better describe what your application intends to do, which can lead to better performance and less surprising driver behavior compared to existing APIs like OpenGL and Direct3D. The ideas behind Vulkan are similar to those of Direct3D 12 and Metal, but Vulkan has the advantage of being cross-platform and allows you to develop for Windows, Linux and Android at the same time (and iOS and macOS via MoltenVK)***.

So what you need to run a Vulkan app:

✅ Vulkan Loader

Dispatches Vulkan API calls to the correct GPU driver.

Installed automatically with:

Vulkan SDK (for developers)

GPU drivers (for users)

✅ GPU Driver with Vulkan Support (ICD)

From your GPU vendor (NVIDIA, AMD, Intel, etc.)

Implements the actual Vulkan functionality.

Defines how commands run on your specific GPU hardware.

**No Extra Runtime = Minimal Dependencies That’s one of the beauties of Vulkan:**



