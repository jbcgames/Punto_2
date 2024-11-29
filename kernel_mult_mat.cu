/**
 * \file kernel.cu
 * \author Ricardo Andrés Velásquez Vélez
 */


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"
#include "cputimer.h"

#include <iostream>
#include <numeric>
#include <random>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#define _IA64_ 1
static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * CUDA kernel that computes dot product values for a given vector pair
 */
__global__ void dotProductKernel_v1(float* A, float* B, float* acum, unsigned size) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        atomicAdd(acum, A[idx] * B[idx]);
    }
}

/**
 * CUDA kernel that computes dot product values for a given vector pair
 */
__global__ void dotProductKernel_v2(float* A, float* B, float* acum, unsigned size)
{
    __shared__ float temp[1024];
    int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int lIdx = threadIdx.x;
    temp[lIdx] = A[gIdx] * B[gIdx];
    __syncthreads();
    int k = 0;
    if (lIdx == 0) {
        for (k = 1; k < blockDim.x; k++) {
            temp[0] += temp[k];
        }
        atomicAdd(acum, temp[0]);
    }
}

/**
 * CUDA kernel that computes dot product values for a given vector pair
 */

__global__ void dotProductKernel_v3(float* A, float* B, float* acum, unsigned size)
{
    __shared__ float temp[1024];
    int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int lIdx = threadIdx.x;
    if (gIdx < size)
        temp[lIdx] = A[gIdx] * B[gIdx];
    else
        temp[lIdx] = 0;

    int k = blockDim.x/2;
    while(k>0) {
        if(lIdx<k){
            temp[lIdx] = temp[lIdx] + temp[lIdx + k];
        }
        k = k/2;
        __syncthreads();
    }
    if (lIdx == 0)
        atomicAdd(acum, temp[0]);
}

/**
 * Host function that copies the data and launches the work on GPU
 */
float gpuDotProduct(float* A, float* B, unsigned size, unsigned numT);

/**
 * Host function to compute the dot product on the CPU
 */
float cpuDotProduct(float* A, float* B, unsigned size);

/**
 * Host function to initialize to random values the float Vectors
 */
void initFloatVec(float* data, unsigned size);



/**
  * 
  **/
void gpuPrintProperties(unsigned GpuID);


int main(void)
{
    unsigned deviceID = 0; ///< Identify wich GPU you want to use in case more than one in a computing system

    CpuTimer cpuTimer1;
    static const int WORK_SIZE = 1<<28; ///< WORK_SIZE defines the size of vectors A and B
    static const int MAX_THREAD = 1024; ///< MAX_THREAD define the number of threas per block
    uint32_t size = MAX_THREAD * (WORK_SIZE / MAX_THREAD + 1);
    float* A = new float[size];
    float* B = new float[size];
    float acumCpu, acumGpu;
    initFloatVec(A, WORK_SIZE);
    initFloatVec(B, WORK_SIZE);
    
    CUDA_CHECK_RETURN(cudaSetDevice(deviceID)); ///< Establish the device to run the kernel
    //gpuPrintProperties(deviceID);               ///< Print some GPU properties

    cpuTimer1.Start();
    acumCpu = cpuDotProduct(A, B, WORK_SIZE);
    cpuTimer1.Stop();

    std::cout << "Vector Size:" << WORK_SIZE << "\nTHREADS_BLOCK:," << MAX_THREAD << "\nCPU_TIME:" << 1000*cpuTimer1.Elapsed().count() << " ms\n";
    acumGpu = gpuDotProduct(A, B, WORK_SIZE, MAX_THREAD);
    std::cout << "ACUM_CPU:" << acumCpu << "\nACUM_GPU:" << acumGpu << std::endl;

    std::cout << "DIFFERENCE:" << acumCpu - acumGpu << std::endl;
    /* Verify the results */

    /* Free memory */
    delete[] A;
    delete[] B;

    return 0;
}

float gpuDotProduct(float* A, float* B, unsigned size, unsigned numT)
{
    float* gpuA;
    float* gpuB;
    float* gpuAcum;
    float acum;

    GpuTimer gpuTimer1 = GpuTimer();
    GpuTimer gpuTimer2 = GpuTimer();

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
    dotProductKernel_v3 <<< blockCount, BLOCK_SIZE >>> (gpuA, gpuB, gpuAcum, size);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    gpuTimer2.Stop();

    CUDA_CHECK_RETURN(cudaMemcpy(&acum, gpuAcum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(gpuA));
    CUDA_CHECK_RETURN(cudaFree(gpuB));
    gpuTimer1.Stop();
    std::cout << "GPU_TIME_TOTAL=" << gpuTimer1.Elapsed() << " ms\nGPU_TIME_KERNEL=" << gpuTimer2.Elapsed() << " ms\n";

    return acum;
}

/**
 * Host function to compute the dot product on the CPU
 */
float cpuDotProduct(float* A, float* B, unsigned size)
{
    float acum = 0;
    for (unsigned i = 0; i < size; i++) {
        acum += A[i] * B[i];
    }
    return acum;
}

/**
*Check the return value of the CUDA runtime API call and exit
* the application if the call has failed.
*/
static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

/*
 * char   name[256];                  //< ASCII string identifying device //
    size_t totalGlobalMem;             //< Global memory available on device in bytes //
    size_t sharedMemPerBlock;          //< Shared memory available per block in bytes //
    int    regsPerBlock;               //< 32-bit registers available per block //
    int    warpSize;                   //< Warp size in threads //
    size_t memPitch;                   //< Maximum pitch in bytes allowed by memory copies //
    int    maxThreadsPerBlock;         //< Maximum number of threads per block //
    int    maxThreadsDim[3];           //< Maximum size of each dimension of a block //
    int    maxGridSize[3];             //< Maximum size of each dimension of a grid //
    int    clockRate;                  //< Clock frequency in kilohertz //
    size_t totalConstMem;              //< Constant memory available on device in bytes //
    int    major;                      //< Major compute capability //
    int    minor;                      //< Minor compute capability //
    size_t textureAlignment;           //< Alignment requirement for textures //
    size_t texturePitchAlignment;      //< Pitch alignment requirement for texture references bound to pitched memory //
    int    deviceOverlap;              //< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. //
    int    multiProcessorCount;        //< Number of multiprocessors on device //
    int    kernelExecTimeoutEnabled;   //< Specified whether there is a run time limit on kernels //
    int    integrated;                 //< Device is integrated as opposed to discrete //
    int    canMapHostMemory;           //< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer //
    int    computeMode;                //< Compute mode (See ::cudaComputeMode) //
    int    maxTexture1D;               //< Maximum 1D texture size //
    int    maxTexture1DMipmap;         //< Maximum 1D mipmapped texture size //
    int    maxTexture1DLinear;         //< Maximum size for 1D textures bound to linear memory //
    int    maxTexture2D[2];            //< Maximum 2D texture dimensions //
    int    maxTexture2DMipmap[2];      //< Maximum 2D mipmapped texture dimensions //
    int    maxTexture2DLinear[3];      //< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory //
    int    maxTexture2DGather[2];      //< Maximum 2D texture dimensions if texture gather operations have to be performed //
    int    maxTexture3D[3];            //< Maximum 3D texture dimensions //
    int    maxTexture3DAlt[3];         //< Maximum alternate 3D texture dimensions //
    int    maxTextureCubemap;          //< Maximum Cubemap texture dimensions //
    int    maxTexture1DLayered[2];     //< Maximum 1D layered texture dimensions //
    int    maxTexture2DLayered[3];     //< Maximum 2D layered texture dimensions //
    int    maxTextureCubemapLayered[2];//< Maximum Cubemap layered texture dimensions //
    int    maxSurface1D;               //< Maximum 1D surface size //
    int    maxSurface2D[2];            //< Maximum 2D surface dimensions //
    int    maxSurface3D[3];            //< Maximum 3D surface dimensions //
    int    maxSurface1DLayered[2];     //< Maximum 1D layered surface dimensions //
    int    maxSurface2DLayered[3];     //< Maximum 2D layered surface dimensions //
    int    maxSurfaceCubemap;          //< Maximum Cubemap surface dimensions //
    int    maxSurfaceCubemapLayered[2];//< Maximum Cubemap layered surface dimensions //
    size_t surfaceAlignment;           //< Alignment requirements for surfaces //
    int    concurrentKernels;          //< Device can possibly execute multiple kernels concurrently //
    int    ECCEnabled;                 //< Device has ECC support enabled //
    int    pciBusID;                   //< PCI bus ID of the device //
    int    pciDeviceID;                //< PCI device ID of the device //
    int    pciDomainID;                //< PCI domain ID of the device //
    int    tccDriver;                  //< 1 if device is a Tesla device using TCC driver, 0 otherwise //
    int    asyncEngineCount;           //< Number of asynchronous engines //
    int    unifiedAddressing;          //< Device shares a unified address space with the host //
    int    memoryClockRate;            //< Peak memory clock frequency in kilohertz //
    int    memoryBusWidth;             //< Global memory bus width in bits //
    int    l2CacheSize;                //< Size of L2 cache in bytes //
    int    maxThreadsPerMultiProcessor;//< Maximum resident threads per multiprocessor //
    int    streamPrioritiesSupported;  //< Device supports stream priorities //
    int    globalL1CacheSupported;     //< Device supports caching globals in L1 //
    int    localL1CacheSupported;      //< Device supports caching locals in L1 //
    size_t sharedMemPerMultiprocessor; //< Shared memory available per multiprocessor in bytes //
    int    regsPerMultiprocessor;      //< 32-bit registers available per multiprocessor //
    int    managedMemory;              //< Device supports allocating managed memory on this system //
    int    isMultiGpuBoard;            //< Device is on a multi-GPU board //
    int    multiGpuBoardGroupID;       //< Unique identifier for a group of devices on the same multi-GPU board //
 * */
void gpuPrintProperties(unsigned GpuID)
{
    cudaDeviceProp prop;
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, GpuID));
    std::cout << "GPU name: " << prop.name << std::endl;
    std::cout << "Total Global Mem: " << prop.totalGlobalMem << std::endl;
    std::cout << "Total Constant Mem: " << prop.totalConstMem << std::endl;
    std::cout << "Shared Mem per Block: " << prop.sharedMemPerBlock << std::endl;
    std::cout << "Register per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warpsize: " << prop.warpSize << std::endl;
    std::cout << "Max threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads Dim: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max Grid Size: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
    std::cout << "Shared Mem per Multiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Register per Multiprocessor: " << prop.regsPerMultiprocessor << std::endl;
    std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Clock Rate: " << prop.clockRate << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << std::endl;
    std::cout << "L2 Cache Size: " << prop.l2CacheSize << std::endl;
    std::cout << "Global L1 cache supported: " << prop.globalL1CacheSupported << std::endl;
    std::cout << "Local L1 cache supported: " << prop.localL1CacheSupported << std::endl;
    std::cout << "Major: " << prop.major << std::endl;
    std::cout << "Minor: " << prop.minor << std::endl;
}

void initFloatVec(float* data, unsigned size)
{
    std::random_device rd;
    for (unsigned i = 0; i < size; ++i) {
        data[i] = (float)((int64_t)rd()  + (int64_t)rd.min() - ((int64_t)rd.max()/2)) / (float)rd.max();
    }
}

