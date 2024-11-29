#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"
#include "cputimer.h"

#define _IA64_ 1
static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

__global__ void dotProductKernel_v1(float* A, float* B, float* acum, unsigned size) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(acum, A[idx] * B[idx]);
    }
}

__global__ void dotProductKernel_v2(float* A, float* B, float* acum, unsigned size) {
    __shared__ float temp[1024];
    int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int lIdx = threadIdx.x;
    temp[lIdx] = A[gIdx] * B[gIdx];
    __syncthreads();
    if (lIdx == 0) {
        for (int k = 1; k < blockDim.x; k++) {
            temp[0] += temp[k];
        }
        atomicAdd(acum, temp[0]);
    }
}

__global__ void dotProductKernel_v3(float* A, float* B, float* acum, unsigned size) {
    __shared__ float temp[1024];
    int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int lIdx = threadIdx.x;
    if (gIdx < size) {
        temp[lIdx] = A[gIdx] * B[gIdx];
    } else {
        temp[lIdx] = 0;
    }

    int k = blockDim.x / 2;
    while (k > 0) {
        if (lIdx < k) {
            temp[lIdx] += temp[lIdx + k];
        }
        k /= 2;
        __syncthreads();
    }
    if (lIdx == 0) {
        atomicAdd(acum, temp[0]);
    }
}

float gpuDotProduct(float* A, float* B, unsigned size, unsigned numT, int kernelVersion);
float cpuDotProduct(float* A, float* B, unsigned size);
void initFloatVec(float* data, unsigned size);
void gpuPrintProperties(unsigned GpuID);
static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err);
float gputimeT, gputimeC;

int main(void) {
    unsigned deviceID = 0;
    static const int WORK_SIZE = 1 << 12;
    std::vector<int> max_thread;

    for (int i = 32; i <= 1024; i = i + 4) {
        max_thread.push_back(i);
    }

    std::ofstream outputFile("Punto2.txt");
    if (!outputFile) {
        std::cerr << "Error creando el archivo Punto2.txt" << std::endl;
        return EXIT_FAILURE;
    }

    outputFile << std::setw(20) << "MAX_THREAD "
               << std::setw(20) << "GPU_TIME_KERNEL_V1"
               << std::setw(20) << "GPU_TIME_KERNEL_V2"
               << std::setw(20) << "GPU_TIME_KERNEL_V3"
               << std::setw(20) << "GPU_TIME_TOTAL" 
               << std::endl;

    for (int MAX_THREAD : max_thread) {
        uint32_t size = WORK_SIZE;
        float* A = new float[size];
        float* B = new float[size];
        float acumGpu;

        initFloatVec(A, WORK_SIZE);
        initFloatVec(B, WORK_SIZE);

        CUDA_CHECK_RETURN(cudaSetDevice(deviceID));

        for (int kernelVersion = 1; kernelVersion <= 3; ++kernelVersion) {
            acumGpu = gpuDotProduct(A, B, WORK_SIZE, MAX_THREAD, kernelVersion);
            outputFile << std::setw(15) << MAX_THREAD
                       << std::setw(20) << ((kernelVersion == 1) ? gputimeT : 0)
                       << std::setw(20) << ((kernelVersion == 2) ? gputimeT : 0)
                       << std::setw(20) << ((kernelVersion == 3) ? gputimeT : 0)
                       << std::setw(20) << gputimeC
                       << std::endl;
        }

        delete[] A;
        delete[] B;

        // Comandos para clonar cambios en el repositorio y hacer push
        std::system("git add Punto2.txt");
        std::string commitCommand = "git commit -m \"Iteración con MAX_THREAD = " + std::to_string(MAX_THREAD) + "\"";
        std::system(commitCommand.c_str());
        std::system("git push");
    }

    outputFile.close();
    return 0;
}

float gpuDotProduct(float* A, float* B, unsigned size, unsigned numT, int kernelVersion) {
    float* gpuA;
    float* gpuB;
    float* gpuAcum;
    float acum;
    GpuTimer gpuTimer1, gpuTimer2;
    
    gpuTimer1.Start();
    CUDA_CHECK_RETURN(cudaMalloc((void**)&gpuA, sizeof(float) * size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&gpuB, sizeof(float) * size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&gpuAcum, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpy(gpuA, A, sizeof(float) * size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(gpuB, B, sizeof(float) * size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemset(gpuAcum, 0, sizeof(float)));

    static const int BLOCK_SIZE = numT;
    const int blockCount = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gpuTimer2.Start();
    switch (kernelVersion) {
        case 1:
            dotProductKernel_v1<<<blockCount, BLOCK_SIZE>>>(gpuA, gpuB, gpuAcum, size);
            break;
        case 2:
            dotProductKernel_v2<<<blockCount, BLOCK_SIZE>>>(gpuA, gpuB, gpuAcum, size);
            break;
        case 3:
            dotProductKernel_v3<<<blockCount, BLOCK_SIZE>>>(gpuA, gpuB, gpuAcum, size);
            break;
        default:
            std::cerr << "Invalid kernel version" << std::endl;
            exit(EXIT_FAILURE);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    gpuTimer2.Stop();

    CUDA_CHECK_RETURN(cudaMemcpy(&acum, gpuAcum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(gpuA));
    CUDA_CHECK_RETURN(cudaFree(gpuB));
    CUDA_CHECK_RETURN(cudaFree(gpuAcum));
    gpuTimer1.Stop();

    gputimeT = gpuTimer2.Elapsed();
    gputimeC = gpuTimer1.Elapsed();

    return acum;
}

float cpuDotProduct(float* A, float* B, unsigned size) {
    float acum = 0;
    for (unsigned i = 0; i < size; ++i) {
        acum += A[i] * B[i];
    }
    return acum;
}

void initFloatVec(float* data, unsigned size) {
    std::random_device rd;
    for (unsigned i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rd()) / static_cast<float>(UINT32_MAX);
    }
}

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess) return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}
