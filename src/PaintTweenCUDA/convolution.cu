/******************************************************************************

This file is part of the source code package for the paper:

Stylizing Animation by Example
by Pierre Benard, Forrester Cole, Michael Kass, Igor Mordatch, James Hegarty, 
Martin Sebastian Senn, Kurt Fleischer, Davide Pesare, Katherine Breeden
Pixar Technical Memo #13-03 (presented at SIGGRAPH 2013)

The original paper, source, and data are available here:

graphics.pixar.com/library/ByExampleStylization/

Copyright (c) 2013 by Disney-Pixar

Permission is hereby granted to use this software solely for 
non-commercial applications and purposes including academic or 
industrial research, evaluation and not-for-profit media
production.  All other rights are retained by Pixar.  For use 
for or in connection with commercial applications and
purposes, including without limitation in or in connection 
with software products offered for sale or for-profit media
production, please contact Pixar at tech-licensing@pixar.com.


******************************************************************************/

#if _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include "convolution.h"
#include "cudaHostUtil.h"
#include <assert.h>

void allocateGaussianKernel(const void* device_name, int kernel_radius, bool normalize)
{
    int kernel_size = 2*kernel_radius + 1;
    // Gaussian kernel
    assert(kernel_size <= BIGGEST_KERNEL_LENGTH);

    float* h_Kernel = new float[kernel_size];
    // Compute gaussian weighting with residualWindowSize = 3 * sigma
    float sigma = (float)(kernel_radius) / 3.0;
    float sigma2 = sigma*sigma;
    for(int i = 0; i <= kernel_radius; i++){
        float dist2 = i*i;
        float weight = expf(-dist2 / (2.f*sigma2));
        if(normalize)
            weight /= (sqrt(2.f * M_PI) * sigma);
        h_Kernel[kernel_radius + i] = weight;
        h_Kernel[kernel_radius - i] = weight;            
    }

    cudaMemcpyToSymbol(device_name, h_Kernel, kernel_size * sizeof(float));
    checkCUDAError("Gaussian kernel initialization");

    delete [] h_Kernel;
}


// Row convolution filter
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4

// Convolution kernel storage in constant memory
__device__ __constant__ float c_Kernel[BIGGEST_KERNEL_LENGTH];
static float c_Kernel_radius = 0;

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void convolutionRowsKernel(
        CudaImageBufferDevice<float> dst,
        //float* dst,
        CudaImageBufferDevice<float> src,
        int imageW, int kernel_radius)
{
    __shared__ float s_Data[ROWS_BLOCKDIM_X + 2*BIGGEST_KERNEL_LENGTH];

    //Current tile and apron limits, relative to row start
    const int tileStart = blockIdx.x * ROWS_BLOCKDIM_X;
    const int tileEnd = tileStart + ROWS_BLOCKDIM_X - 1;
    const int apronStart = tileStart - kernel_radius;
    const int apronEnd = tileEnd   + kernel_radius;

    //Clamp tile and apron limits by image borders
    const int tileEndClamped = min(tileEnd, imageW - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int apronEndClamped = min(apronEnd, imageW - 1);

    //Row start index in src
    const int rowStart = blockIdx.y;

    const int apronStartAligned = tileStart - kernel_radius;

    const int loadPos = apronStartAligned + threadIdx.x;
    //Set the entire data cache contents
    //Load global memory values, if indices are within the image borders,
    //or initialize with zeroes otherwise
    if(loadPos >= apronStart)
    {
        const int smemPos = loadPos - apronStart;

        s_Data[smemPos] = ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ? src.pixel(loadPos, rowStart) : 0;
    }

    //Compute and store results
    __syncthreads();

    const int writePos = tileStart + threadIdx.x;
    if(writePos <= tileEndClamped){
        const int smemPos = kernel_radius + threadIdx.x;//writePos - apronStart;
        float sum = 0.0;

#pragma unroll
        for(int j = -kernel_radius; j <= kernel_radius; j++)
            sum += c_Kernel[kernel_radius - j] * s_Data[smemPos + j];

        dst.setPixel(sum, writePos, rowStart);
    }
}

void convolutionRowsGPU(CudaImageBufferHost<float>& h_Dst,
                        CudaImageBufferHost<float>& h_Src,
                        int level, int kernel_radius)
{
    if (c_Kernel_radius != kernel_radius) {
        allocateGaussianKernel(c_Kernel, kernel_radius, true);
        c_Kernel_radius = kernel_radius;
    }
    dim3 blocks(iDivUp(h_Src.width(level), ROWS_BLOCKDIM_X), h_Src.height(level));
    dim3 threads(2*kernel_radius + ROWS_BLOCKDIM_X);

    convolutionRowsKernel<<<blocks, threads>>>(h_Dst, h_Src, h_Src.width(level),kernel_radius);
}

// Column convolution filter
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8

__global__ void convolutionColumnsKernel(
        CudaImageBufferDevice<float> dst,
        CudaImageBufferDevice<float> src,
        int imageH, int kernel_radius)
{
    //Data cache
    __shared__ float data[COLUMNS_BLOCKDIM_X * (2 * BIGGEST_KERNEL_LENGTH + COLUMNS_BLOCKDIM_Y)];

    //Current tile and apron limits, in rows
    const int tileStart = blockIdx.y * COLUMNS_BLOCKDIM_Y;
    const int tileEnd = tileStart + COLUMNS_BLOCKDIM_Y - 1;
    const int apronStart = tileStart - kernel_radius;
    const int apronEnd = tileEnd + kernel_radius;

    //Clamp tile and apron limits by image borders
    const int tileEndClamped = min(tileEnd, imageH - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int apronEndClamped = min(apronEnd, imageH - 1);

    //Current column index
    const int columnStart = (blockIdx.x * COLUMNS_BLOCKDIM_X) + threadIdx.x;

    //Shared and global memory indices for current column
    int smemPos = (threadIdx.y * COLUMNS_BLOCKDIM_X) + threadIdx.x;
    //Cycle through the entire data cache
    //Load global memory values, if indices are within the image borders,
    //or initialize with zero otherwise
    for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
        data[smemPos] = ((y >= apronStartClamped) && (y <= apronEndClamped)) ? src.pixel(columnStart,y) : 0;
        smemPos += COLUMNS_BLOCKDIM_X*COLUMNS_BLOCKDIM_Y;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    //Shared and global memory indices for current column
    smemPos = (threadIdx.y + kernel_radius) * COLUMNS_BLOCKDIM_X + threadIdx.x;
    //Cycle through the tile body, clamped by image borders
    //Calculate and output the results
    for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
        float sum = 0.0;

#pragma unroll
        for(int k = -kernel_radius; k <= kernel_radius; k++)
            sum += data[smemPos + (k * COLUMNS_BLOCKDIM_X)] * c_Kernel[kernel_radius - k];

        dst.setPixel(sum,columnStart,y);
        smemPos += COLUMNS_BLOCKDIM_X*COLUMNS_BLOCKDIM_Y;
    }
}

void convolutionColumnsGPU(CudaImageBufferHost<float>& h_Dst,
                           CudaImageBufferHost<float>& h_Src,
                           int level, int kernel_radius)
{
    if (c_Kernel_radius != kernel_radius) {
        allocateGaussianKernel(c_Kernel, kernel_radius, true);
        c_Kernel_radius = kernel_radius;
    }
    dim3 blocks(iDivUp(h_Src.width(level), COLUMNS_BLOCKDIM_X),
                iDivUp(h_Src.height(level), COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel<<<blocks, threads>>>(h_Dst, h_Src, h_Src.height(level),kernel_radius);
}


template <class T, unsigned int blockSize>
__device__ void warpReduce(volatile T *sdata, unsigned int tid) {
    if (blockSize >=  64)
        sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    if (blockSize >=  32)
        sdata[tid] = max(sdata[tid], sdata[tid + 16]);
    if (blockSize >=  16)
        sdata[tid] = max(sdata[tid], sdata[tid +  8]);
    if (blockSize >=    8)
        sdata[tid] = max(sdata[tid], sdata[tid +  4]);
    if (blockSize >=    4)
        sdata[tid] = max(sdata[tid], sdata[tid +  2]);
    if (blockSize >=    2)
        sdata[tid] = max(sdata[tid], sdata[tid +  1]);
}

extern "C"
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}


template <class T, unsigned int blockSize>
__global__ void reduceMax(CudaImageBufferDevice<T> g_idata, T *g_odata, unsigned int n, int level, int minIndex) {
    extern __shared__ T sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid + minIndex;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T myMax = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n + minIndex)
    {
        myMax = max(myMax, g_idata.at(i,level));
        // ensure we don't read out of bounds
        if (i + blockSize < n + minIndex)
            myMax = max(myMax, g_idata.at(i+blockSize,level));

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = myMax;


    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = myMax = max(myMax, sdata[tid + 256]);
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = myMax = max(myMax, sdata[tid + 128]);
        } __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid <   64)
        {
            sdata[tid] = myMax = max(myMax, sdata[tid + 64]);
        }
        __syncthreads();
    }
    if (tid < 32)
        warpReduce<T,blockSize>(sdata, tid);

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

float bufferMax(CudaImageBufferHost<float>& d_idata, int level, int styleIndex, int numStyles)
{
    int size = d_idata.width(level)*d_idata.height(level)/numStyles;

    int maxThreads = 256;  // number of threads per block
    int threads = (size < maxThreads*2) ? nextPow2((size + 1)/ 2) : maxThreads;
    int blocks  = (size + (threads * 2 - 1)) / (threads * 2);
    blocks = (blocks > 64) ? 64 : blocks;

    float* d_odata = NULL;
    cudaMalloc((void**) &d_odata, blocks*sizeof(float));

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    int minIndex = floorf( (float)(styleIndex * size) );

    switch (threads)
    {
    case 512:
        reduceMax<float, 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, level, minIndex); break;
    case 256:
        reduceMax<float, 256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, level, minIndex); break;
    case 128:
        reduceMax<float, 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, level, minIndex); break;
    case 64:
        reduceMax<float, 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, level, minIndex); break;
    case 32:
        reduceMax<float, 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, level, minIndex); break;
    case 16:
        reduceMax<float, 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, level, minIndex); break;
    case  8:
        reduceMax<float, 8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, level, minIndex); break;
    case  4:
        reduceMax<float, 4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, level, minIndex); break;
    case  2:
        reduceMax<float, 2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, level, minIndex); break;
    case  1:
        reduceMax<float, 1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, level, minIndex); break;
    }

    float* h_odata = (float*) malloc(blocks*sizeof(float));
    cudaMemcpy( h_odata, d_odata, blocks*sizeof(float), cudaMemcpyDeviceToHost);

    float result = 0.0;

    for(int i=0; i<blocks; i++)
    {
        result = max(result, h_odata[i]);
    }

    free(h_odata);
    cudaFree(d_odata);

    return result;
}


