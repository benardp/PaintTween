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

#ifndef _TEXSYNTH_KERNEL_H_
#define _TEXSYNTH_KERNEL_H_

#if _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <stdio.h>
#include <assert.h>
#include "texSynth_kernel.h"
#include "texSynth_util.h"
#include "texSynth_interface.h"
#include "cudaImagePyramid.h"
#include "cudaImageBuffer.h"
#include "cudaTexture.h"

#include "types.h"
#include "convolution.h"

// the number of pixels in each threadblock. This parameter can affect performance,
// but the default of 16 is probably fine.
const int kBlockSizeX = 16;
const int kBlockSizeY = 8;
const dim3 kThreadsPerBlock(kBlockSizeX, kBlockSizeY);
// Maximum number of blocks in a kernel launch. If this is too large, the kernel
// can time out.
const int kMaxBlocksX = 32;
const int kMaxBlocksY = 32;

/////////////////////////////////////////////////////////////////////////
// Macro definitions for the image pyramids. The macro defines a
// CudaImagePyramidHost, a texture<> reference, and a device-side offsets array.
// The structure is cumbersome because you can't pass texture<> references
// as function arguments (as of CUDA 4.0). 
/////////////////////////////////////////////////////////////////////////
CUDA_IMAGE_PYRAMID_LAYERED(float4, gExemplarBase)
CUDA_IMAGE_PYRAMID_LAYERED(float4, gExemplarOutput)
CUDA_IMAGE_PYRAMID_LAYERED(float, gExemplarOrientation)
CUDA_IMAGE_PYRAMID_LAYERED(float, gExemplarDistanceTransform)
CUDA_IMAGE_PYRAMID_LAYERED(int, gExemplarObjectIDs)

CUDA_IMAGE_PYRAMID_LAYERED(float4, gCumulFieldB)
CUDA_IMAGE_PYRAMID_LAYERED(float4, gCumulFieldF)

CUDA_IMAGE_PYRAMID_LAYERED(float4, gImageBase)
CUDA_IMAGE_PYRAMID(float, gImageOrientation)
CUDA_IMAGE_PYRAMID_LAYERED(float, gImageDistanceTransform)
CUDA_IMAGE_PYRAMID(float2, gImageScale)

CUDA_IMAGE_PYRAMID(float4, gRandom)

CUDA_IMAGE_PYRAMID(int, gImageId)
CUDA_IMAGE_PYRAMID(float2, gImageSurfaceId)
CUDA_IMAGE_PYRAMID_LAYERED(float2, gImageVelB)
CUDA_IMAGE_PYRAMID_LAYERED(float2, gImageVelF)

CUDA_IMAGE_PYRAMID(float4, gGuideF)
CUDA_IMAGE_PYRAMID(float4, gGuideB)

CUDA_IMAGE_PYRAMID(float4, gFrameToKeyRibbonF)
CUDA_IMAGE_PYRAMID(float4, gFrameToKeyRibbonB)

// Convolution kernel storage in constant memory
__device__ __constant__ float gGaussianKernel[BIGGEST_KERNEL_LENGTH];
static float gGaussianKernelSize = 0;

__device__ __constant__ TsParameters gParams;
__device__ __constant__ int gCurrentBuffer = 0;

struct S
{
	float one;
	int two;
	bool three;
};

__device__ __constant__ S d_s;

static int _currentBuffer = 0;

void TexSynth::setup(TsParameters params)
{
    if (gGaussianKernelSize != params.residualWindowSize) {
        allocateGaussianKernel(gGaussianKernel, params.residualWindowSize, false);
        gGaussianKernelSize = params.residualWindowSize;
    }
    uploadParameters(params);
    cudaMemcpyToSymbol(gCurrentBuffer, &_currentBuffer, sizeof(int));
}

void TexSynth::uploadParameters(const TsParameters &params)
{
    cudaMemcpyToSymbol(gParams, &params, sizeof(TsParameters));
    checkCUDAError("Copy params to constant memory");
}

__host__ const void* TexSynth::getTextureReferenceByName( const char *name) 
{
	if( strcmp( name, "gExemplarBase_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gExemplarBase_TEXTURE_PYRAMID;
	else if( strcmp( name, "gExemplarOutput_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gExemplarOutput_TEXTURE_PYRAMID;
	else if( strcmp( name, "gExemplarOrientation_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gExemplarOrientation_TEXTURE_PYRAMID;
	else if( strcmp( name, "gExemplarDistanceTransform_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gExemplarDistanceTransform_TEXTURE_PYRAMID;
	else if( strcmp( name, "gExemplarObjectIDs_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gExemplarObjectIDs_TEXTURE_PYRAMID;
	else if( strcmp( name, "gCumulFieldB_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gCumulFieldB_TEXTURE_PYRAMID;
	else if( strcmp( name, "gCumulFieldF_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gCumulFieldF_TEXTURE_PYRAMID;
	else if( strcmp( name, "gImageBase_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gImageBase_TEXTURE_PYRAMID;
	else if( strcmp( name, "gImageOrientation_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gImageOrientation_TEXTURE_PYRAMID;
	else if( strcmp( name, "gImageDistanceTransform_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gImageDistanceTransform_TEXTURE_PYRAMID;
	else if( strcmp( name, "gImageScale_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gImageScale_TEXTURE_PYRAMID;
	else if( strcmp( name, "gRandom_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gRandom_TEXTURE_PYRAMID;
	else if( strcmp( name, "gImageId_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gImageId_TEXTURE_PYRAMID;
	else if( strcmp( name, "gImageSurfaceId_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gImageSurfaceId_TEXTURE_PYRAMID;
	else if( strcmp( name, "gImageVelB_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gImageVelB_TEXTURE_PYRAMID;
	else if( strcmp( name, "gImageVelF_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gImageVelF_TEXTURE_PYRAMID;
	else if( strcmp( name, "gGuideF_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gGuideF_TEXTURE_PYRAMID;
	else if( strcmp( name, "gGuideB_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gGuideB_TEXTURE_PYRAMID;
	else if( strcmp( name, "gFrameToKeyRibbonF_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gFrameToKeyRibbonF_TEXTURE_PYRAMID;
	else if( strcmp( name, "gFrameToKeyRibbonB_TEXTURE_PYRAMID") == 0 )
		return (const void*) &gFrameToKeyRibbonB_TEXTURE_PYRAMID;
	else return 0;
}

//#define DEBUG_CULLING

#ifdef DEBUG_CULLING
__device__ int gCulledCount = 0;
#endif

__device__ bool isImagePixelCulled(int x, int y, float distFudge = 0)
{
    float dist = PYRAMID_FETCH_LAYER(gImageDistanceTransform, x, y, gCurrentBuffer);
    bool culled = dist > (gParams.distTransCull - distFudge);
#ifdef DEBUG_CULLING
    if (culled) {
        atomicAdd(&gCulledCount, 1);
    }
#endif
    return culled;
}

__device__ bool isImagePixelCulledPreviousFrame(int x, int y, int level, float distFudge = 0)
{
    float dist = PYRAMID_FETCH_LAYER(gImageDistanceTransform, x, y, (gCurrentBuffer+1)%2);
    bool culled = dist > (gParams.distTransCull - distFudge);
#ifdef DEBUG_CULLING
    if (culled) {
        atomicAdd(&gCulledCount, 1);
    }
#endif
    return culled;
}

__device__ bool isExemplarPixelCulled(int x, int y, int level, int layer)
{
    float dist = PYRAMID_FETCH_LAYER(gExemplarDistanceTransform, x, y, layer);
    bool culled = dist > gParams.distTransCull * 1.25;
    return culled;
}

__device__ float4 randomFloat4(int x, int y, float4& fractional)
{
    float4 random = PYRAMID_FETCH(gRandom, x, y);
    float4 intpart = floorf(random);
    fractional = random - intpart;
    return random;
}

template <class T>
__device__ float borderFeather(CudaImageBufferDevice<T> buffer, int x, int y, int level, int width)
{
    float border = width >> level;
    float xdist = min(x + 0.5, buffer.width(level) - x + 0.5);
    float ydist = min(y + 0.5, buffer.height(level) - y + 0.5);
    float dist = min(xdist, ydist);
    float feather = clamp(dist / border, 0.f, 1.f);
    return feather;
}

template <class T>
__device__ bool advectionBorderDither(CudaImageBufferDevice<T> buffer, int x, int y, int level, float random)
{
    float feather = borderFeather(buffer, x, y, level, 32);
    return feather < random*0.99;
}

template <class T>
__device__ bool keyframeBorderDither(CudaImageBufferDevice<T> buffer, int x, int y, int level)
{
    float4 random;
    randomFloat4(x,y,random);
    float feather = borderFeather(buffer, x, y, level, 64);
    return feather < random.x*0.49 + 0.5;
}

// calculate histogram of offsets
template <class T>
__global__ void calculateOffsetsHistogram(int level, CudaImageBufferDevice<PatchXF> offsets,
                                          CudaImageBufferDevice<T> outputa){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(!offsets.inBounds(x,y,level) ||
            isImagePixelCulled(x,y))
        return;

    PatchXF offset = offsets.pixel(x,y);

    if(!outputa.inBounds(offset.x,offset.y,level))
        return;

    //outputa.atomicAddition(1.0, offset.x, offset.y);

}

// copy pixels from the output style to outputa
__global__ void offsetsToOutput(int level, CudaImageBufferDevice<PatchXF> offsets, 
                                CudaImageBufferDevice<Color4> outputa){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(!outputa.inBounds(x,y,level))
        return;

    PatchXF offset = offsets.pixel(x,y);
    Color4 output = toColor4(PYRAMID_FETCH_LAYER(gExemplarOutput, offset.x, offset.y, offset.layer),offset.luminanceShift);
    outputa.setPixel(output, x, y);
}

// copy pixels from the output style to outputa
__global__ void offsetsToInput(int level, CudaImageBufferDevice<PatchXF> offsets,
                                CudaImageBufferDevice<Color4> outputa){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(!outputa.inBounds(x,y,level))
        return;

    PatchXF offset = offsets.pixel(x,y);
    Color4 output = toColor4(PYRAMID_FETCH_LAYER(gExemplarBase, offset.x, offset.y, offset.layer),offset.luminanceShift);
    outputa.setPixel(output, x, y);
}

__global__ void cumulativeAdvectionFieldToOutput(int level,
                                                 CudaImageBufferDevice<RibbonP> cumulativeAdvectionField,
                                                 CudaImageBufferDevice<Color4> outputa,
                                                 CudaImageBufferDevice<Color4> keyFrame){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(!outputa.inBounds(x,y,level))
        return;

    RibbonP field = cumulativeAdvectionField.pixel(x,y);
    Color4 output(0);
    if(field.isValid())
        output = keyFrame.pixel(x + field.x, y + field.y);
    outputa.setPixel(output, x, y);
}

// copy pixels from the style input to outputa
__global__ void offsetsToBase(int level, CudaImageBufferDevice<PatchXF> offsets, 
                              CudaImageBufferDevice<Color4> outputa){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(!outputa.inBounds(x,y,level))
        return;

    PatchXF offset = offsets.pixel(x,y);

    Color4 output = toColor4(PYRAMID_FETCH_LAYER(gExemplarBase, offset.x, offset.y, offset.layer));
    outputa.setPixel(output, x, y);
}

// copy pixels from the style input to outputa
__global__ void offsetsToDistTrans(int level, CudaImageBufferDevice<PatchXF> offsets,
                                   CudaImageBufferDevice<float2> outputa){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(!outputa.inBounds(x,y,level))
        return;

    PatchXF offset = offsets.pixel(x,y);

    float e = PYRAMID_FETCH_LAYER(gExemplarDistanceTransform, offset.x, offset.y, offset.layer);
    float i = PYRAMID_FETCH_LAYER(gImageDistanceTransform, x, y, gCurrentBuffer);
    float2 output = make_float2(i,e);
    outputa.setPixel(output, x, y);
}

__global__ void cacheVelocities(
        int level,
        CudaImageBufferDevice<float2> velF,
        CudaImageBufferDevice<float2> velB,
        bool previousForward = false,
        bool previousBackward = false)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(!velF.inBounds(x,y,level))
        return;

    int backwardbuf = (gCurrentBuffer + (previousBackward) ? 1 : 0) % 2;
    int forwardbuf = (gCurrentBuffer + (previousForward) ? 1 : 0) % 2;

    float2 vf = PYRAMID_FETCH_LAYER(gImageVelF, x, y, forwardbuf);
    float2 vb = PYRAMID_FETCH_LAYER(gImageVelB, x, y, backwardbuf);

    velF.setPixel(vf,x,y);
    velB.setPixel(vb,x,y);
}

__global__ void cacheRibbon(
        int level,
        CudaImageBufferDevice<RibbonP> ribbonF,
        CudaImageBufferDevice<RibbonP> ribbonB)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(!ribbonF.inBounds(x,y,level))
        return;

    ribbonF.setPixel(toRibbonP(PYRAMID_FETCH(gFrameToKeyRibbonF, x, y)), x, y);
    ribbonB.setPixel(toRibbonP(PYRAMID_FETCH(gFrameToKeyRibbonB, x, y)), x, y);
}

__device__ float coherenceTerm(
        CudaImageBufferDevice<PatchXF> offsets,
        int x, int y, int i, int j,
        float2 testCenter, float2 test, int level){

    PatchXF canvasOffset = offsets.pixel(x+i,y+j);

    // the offset at i,j,k (following the velocity path)
    // location of pixel in style output
    float2 canvas = canvasOffset.xy();

    // normalize these distances by the pyramid level so that the magnitude of the
    // penalty remains constant when upsampling.
    float distExemplar = length2(canvas - testCenter) * (1 << level);

    float distImage = float(i*i + j*j) * (1 << level);

    // want the length ratio between image distance and exemplar coordinate distance to be 1
    float ratio = clamp(distExemplar / distImage, 1.0f / gParams.coherenceRatioRange, gParams.coherenceRatioRange);
    float logratio = logf(ratio);

    // normalization factor
    float Z = logf(gParams.coherenceRatioRange) * logf(gParams.coherenceRatioRange);
    float Eratio = (logratio * logratio) / Z;

    // want the angle between image and exemplar to be 0
    // exTest is the location in exemplar space of (x+i,y+j) relative to testOffset
    float2 dirTest = test - testCenter;
    float lenTest = length(dirTest);

    dirTest.x = (lenTest > 1e-4f) ? dirTest.x * (1.0f / lenTest) : 1.0;
    dirTest.y = (lenTest > 1e-4f) ? dirTest.y * (1.0f / lenTest) : 0.0;

    float2 dirCanvas = canvas - testCenter;
    float lenCanvas = length(dirCanvas);

    dirCanvas.x = (lenCanvas > 1e-4f) ? dirCanvas.x * (1.0f / lenCanvas) : 1.0;
    dirCanvas.y = (lenCanvas > 1e-4f) ? dirCanvas.y * (1.0f / lenCanvas) : 0.0;

    float dirTestDotDirCanvas = dot(dirTest,dirCanvas);
    float angle = clamp( acosf(dirTestDotDirCanvas), -1.0f, +1.0f );

    float Eangle = min(angle, gParams.coherenceAngleRange * M_PI) / (gParams.coherenceAngleRange * M_PI);
    // use the product of errors, to prevent error from growing when either ratio or
    // angle error goes outside the limit
    return (1.0f - (1.0f-Eratio) * (1.0f-Eangle));
}

inline __device__ float screenToExemplarAngle(float x, float y, float ex, float ey, int layer){
    if(gParams.orientationEnabled){
        float theta = PYRAMID_FETCH_LAYER(gExemplarOrientation, ex, ey, layer) -
                PYRAMID_FETCH(gImageOrientation,x,y);
        if(theta > M_PI_2)
            return theta - M_PI;
        if(theta < -M_PI_2)
            return M_PI + theta;
        return theta;
    }
    return 0.0;
}

__device__ inline float timeDerivativeTerms(
        AdvectedBuffersDevice advected, PatchXF prevOffset,
        float2 prevBasis0, float2 prevBasis1,
        Color4 base, Color4 testBase,
        Color4 canvasOutput, Color4 testOutput,
        int x, int y, int i, int j)
{
    Color4 prevBase = advected.base.pixel(x+i, y+j);
    Color4 prevOutput = advected.output.pixel(x+i, y+j);

    float2 prevTest;
    calculateNewExemplarCoord((float)i,(float)j, prevOffset.xy(), prevBasis0, prevBasis1, &prevTest);

    Color4 prevExemplarBase = toColor4(PYRAMID_FETCH_LAYER(gExemplarBase, prevTest.x, prevTest.y, prevOffset.layer));
    Color4 prevExemplarOutput = toColor4(PYRAMID_FETCH_LAYER(gExemplarOutput, prevTest.x, prevTest.y, prevOffset.layer));

    if(gParams.colorspace == 1){
        prevBase = RGBtoLAB(prevBase);
        prevOutput = RGBtoLAB(prevOutput);
        prevExemplarBase = RGBtoLAB( prevExemplarBase);
        prevExemplarOutput = RGBtoLAB(prevExemplarOutput);
    }

    float residual = sumSquaredDiff(base - prevBase, (testBase - prevExemplarBase)) * gParams.timeDerivativeInputWeight;
    residual += sumSquaredDiff(canvasOutput - prevOutput, (testOutput - prevExemplarOutput)) * gParams.timeDerivativeOutputWeight;
    residual += sumSquaredDiff(prevOutput, testOutput) * gParams.temporalCoherenceWeight;

    //float time_step = max(advected.timeStep.pixel(x+i,y+j), 1);
    //residual /= time_step;

    return residual;
}

__device__ inline float distTransformTerm(int x, int y, int layer, float2 test, float scaleV)
{
    // distance transform
    float distExemplar = PYRAMID_FETCH_LAYER(gExemplarDistanceTransform, test.x, test.y, layer);
    // Adjust the exemplar's distance transform value to match the scaled neighborhood size.
    distExemplar /= scaleV;
    float distImage = PYRAMID_FETCH_LAYER(gImageDistanceTransform, x, y, gCurrentBuffer);

    float diff = abs(distExemplar - distImage);
    // Don't let the distance transform difference become too huge.
    diff = clamp(diff, 0.0f, 10.0f);

    return diff * gParams.distTransWeight;
}

__device__ inline float histogramTerm(CudaImageBufferDevice<float> histogram,
                                      int level, int layer, float2 test)
{
    float style_height = histogram.height(level) / gParams.numStyles;
    // number of time this style's pixel has been used in the previous frame
    float histo = histogram.pixel(test.x,test.y + style_height*layer);
    float histoWeight = (histo > gParams.offsetsHistogramThreshold[layer]) ?
                (histo - gParams.offsetsHistogramThreshold[layer]) * gParams.offsetsHistogramSlope[layer] : 0.0;
    return histoWeight;
}

__device__ inline float objectIdTerm(int level, int x, int y, int layer, float2 test)
{
    int imgID = PYRAMID_FETCH(gImageId, x, y);
    int styleID = PYRAMID_FETCH_LAYER(gExemplarObjectIDs, test.x, test.y, layer);
    if (imgID != styleID)
        return gParams.styleObjIdWeight;
    return 0.f;
}

__device__ float hysteresisTerm(int x, int y, int level)
{
    float maxD = gParams.maxDistortion / (float)powf(2,level);
    float2 delta_xy_00;
    float2 delta_xy_10;
    float2 delta_xy_01;
    float2 delta_xy_11;

    if(gParams.direction == FORWARD){
        delta_xy_00 = PYRAMID_FETCH_LAYER(gImageVelB, x, y, gCurrentBuffer);
        delta_xy_10 = PYRAMID_FETCH_LAYER(gImageVelB, x+1.f, y, gCurrentBuffer);
        delta_xy_01 = PYRAMID_FETCH_LAYER(gImageVelB, x, y+1.f, gCurrentBuffer);
        delta_xy_11 = PYRAMID_FETCH_LAYER(gImageVelB, x+1.f, y+1.f, gCurrentBuffer);
    }else{
        delta_xy_00 = PYRAMID_FETCH_LAYER(gImageVelF, x, y, gCurrentBuffer);
        delta_xy_10 = PYRAMID_FETCH_LAYER(gImageVelF, x+1.f, y, gCurrentBuffer);
        delta_xy_01 = PYRAMID_FETCH_LAYER(gImageVelF, x, y+1.f, gCurrentBuffer);
        delta_xy_11 = PYRAMID_FETCH_LAYER(gImageVelF, x+1.f, y+1.f, gCurrentBuffer);
    }

    float maxX = max(delta_xy_00.x,max(delta_xy_01.x,max(delta_xy_10.x,delta_xy_11.x)));
    float maxY = max(delta_xy_00.y,max(delta_xy_01.y,max(delta_xy_10.y,delta_xy_11.y)));
    float minX = min(delta_xy_00.x,min(delta_xy_01.x,min(delta_xy_10.x,delta_xy_11.x)));
    float minY = min(delta_xy_00.y,min(delta_xy_01.y,min(delta_xy_10.y,delta_xy_11.y)));
    // distortion test
    if( (maxX - minX) < maxD && (maxY - minY) < maxD){
        return gParams.hysteresisWeight; // 2.0 for muntz
    }
    return 0.f;
}

__device__ float calculateResidual(
        CudaImageBufferDevice<PatchXF> offsets,
        CudaImageBufferDevice<Color4> canvasOutputCache,
        CudaImageBufferDevice<float> histogram,
        AdvectedBuffersDevice advectedF,
        AdvectedBuffersDevice advectedB,
        int x, int y, int level, PatchXF testEi)
{
    PatchXF prevOffsetF = advectedF.offsets.pixel(x, y);
    PatchXF prevOffsetB = advectedB.offsets.pixel(x, y);
    bool centerPrevOffsetValid = advectedF.offsets.pixel(x, y).hysteresis > 0;
    bool centerNextOffsetValid = advectedB.offsets.pixel(x, y).hysteresis > 0;

    float2 fBasis0, fBasis1, bBasis0, bBasis1;
    float angle = screenToExemplarAngle(x, y, prevOffsetF.x, prevOffsetF.y, prevOffsetF.layer);
    screenToExemplarBasis(angle, gParams.residualWindowScaleU * prevOffsetF.scaleU, gParams.residualWindowScaleV * prevOffsetF.scaleV, &fBasis0, &fBasis1);
    angle = screenToExemplarAngle(x, y, prevOffsetB.x, prevOffsetB.y, prevOffsetB.layer);
    screenToExemplarBasis(angle, gParams.residualWindowScaleU * prevOffsetB.scaleU, gParams.residualWindowScaleV * prevOffsetB.scaleV, &bBasis0, &bBasis1);

    float sumWeights = 0.f;
    float sumResidual = 0.f;

    if(centerPrevOffsetValid){
        sumResidual -= hysteresisTerm(x,y,level);
    }

    for(int i = -gParams.residualWindowSize; i <= gParams.residualWindowSize; i++){
        for(int j = -gParams.residualWindowSize; j <= gParams.residualWindowSize; j++){
            float residual = 0;

            if(!offsets.inBounds(x+i, y+j, level)){
                continue;
            }

            // (x+i,y+j) - location in image (canvas)
            // (test.x, test.y) - location in style
            Color4 base = toColor4(PYRAMID_FETCH_LAYER(gImageBase, x + i, y + j, gCurrentBuffer));
            Color4 canvasOutput = canvasOutputCache.pixel(x+i, y+j);
            if(gParams.colorspace == 1){
                base = RGBtoLAB(base);
                canvasOutput = RGBtoLAB(canvasOutput);
            }

            bool prevOffsetValid = centerPrevOffsetValid && advectedF.offsets.pixel(x+i, y+j).hysteresis > 0;
            bool nextOffsetValid = centerNextOffsetValid && advectedB.offsets.pixel(x+i, y+j).hysteresis > 0;

            Color4 testBase;
            Color4 testOutput;
            float2 test;
            float2 eBasis0, eBasis1;
            screenToExemplarBasis(testEi.theta, gParams.residualWindowScaleU * testEi.scaleU, gParams.residualWindowScaleV * testEi.scaleV, &eBasis0, &eBasis1);
            // calculate where "i,j" would be in the angle-skewed window of the exemplar centered at testExi, testEyi
            calculateNewExemplarCoord((float)i,(float)j, testEi.xy(), eBasis0, eBasis1, &test);
            if(test.x >= histogram.width(level)){
                test.x -= histogram.width(level) ;
            }else if(test.x < 0){
                test.x += histogram.width(level);
            }
            if(test.y >= histogram.height(level)){
                test.y -= histogram.height(level) ;
            }else if(test.y < 0){
                test.y += histogram.height(level);
            }
            testBase = toColor4(PYRAMID_FETCH_LAYER(gExemplarBase, test.x, test.y, testEi.layer));
            testOutput = toColor4(PYRAMID_FETCH_LAYER(gExemplarOutput, test.x, test.y, testEi.layer));

            if(gParams.colorspace == 1){
                testBase = RGBtoLAB(testBase);
                testOutput = RGBtoLAB(testOutput);
            }

            residual += sumSquaredDiff(base,testBase,gParams.alphaWeight) * gParams.inputAnalogyWeight;

            // Time derivative
            if((gParams.direction == FORWARD || gParams.direction == BIDIRECTIONAL) &&
                    prevOffsetValid && (gParams.timeDerivativeInputWeight > 0.f || gParams.timeDerivativeOutputWeight > 0.f || gParams.temporalCoherenceWeight > 0.f))
            {
                residual += timeDerivativeTerms(
                            advectedF, prevOffsetF, fBasis0, fBasis1,
                            base, testBase, canvasOutput, testOutput,
                            x, y, i, j);
            }

            if((gParams.direction == BACKWARD || gParams.direction == BIDIRECTIONAL) &&
                    nextOffsetValid && (gParams.timeDerivativeInputWeight > 0.f || gParams.timeDerivativeOutputWeight > 0.f || gParams.temporalCoherenceWeight > 0.f))
            {
                residual += timeDerivativeTerms(
                            advectedB, prevOffsetB, bBasis0, bBasis1,
                            base, testBase, canvasOutput, testOutput,
                            x, y, i, j);
            }

            if(i==0 && j==0){
                if(gParams.distTransWeight > 0.f)
                    residual += distTransformTerm(x, y, testEi.layer, test, testEi.scaleV);

                // Histogram penalty
                residual += histogramTerm(histogram, level, testEi.layer, test);

                // incorporate style object ids:
                if (gParams.styleObjIdWeight > 0.f)
                    residual += objectIdTerm(level, x, y, testEi.layer, test);
            }else{
                residual += sumSquaredDiff(canvasOutput, testOutput,gParams.alphaWeight) * gParams.canvasOutputWeight;

                // coherence term
                if (gParams.coherenceWeight > 0.f) {
                    float c = coherenceTerm(offsets, x, y, i, j, testEi.xy(), test, level);
                    residual += c * gParams.coherenceWeight;
                }
            }

            float weight = gGaussianKernel[i + gParams.residualWindowSize] *
                    gGaussianKernel[j + gParams.residualWindowSize];
            sumResidual += residual * weight;
            sumWeights += weight;
        }
    }
    if(sumWeights < EPSILON)
        return 10e30;

    return (sumResidual / sumWeights);
}

__global__ void updateResidualCache_kernel( 
        CudaImageBufferDevice<PatchXF> offsets,
        CudaImageBufferDevice<Color4> canvasOutputCache,
        CudaImageBufferDevice<float> histogram,
        CudaImageBufferDevice<float> outputa,
        AdvectedBuffersDevice advectedF,
        AdvectedBuffersDevice advectedB,
        int level)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!outputa.inBounds(x,y,level)) {
        return;
    }

    if (isImagePixelCulled(x,y)) {
        outputa.setPixel(0.0f, x, y);
        return;
    }

    PatchXF offset = offsets.pixel(x,y);

    float output = calculateResidual(
                offsets,
                canvasOutputCache,
                histogram,
                advectedF, advectedB,
                x, y, level, offset);

    outputa.setPixel(output, x, y);
}

inline __device__ float test_offset(
        CudaImageBufferDevice<PatchXF> offsets,
        CudaImageBufferDevice<Color4> canvasOutputCache,
        CudaImageBufferDevice<float> histogram,
        AdvectedBuffersDevice advectedF,
        AdvectedBuffersDevice advectedB,
        int x, int y, PatchXF test, int level, float image_alpha)
{
    if(!isExemplarPixelCulled(test.x,test.y,level,test.layer)) {

        if(!gParams.transparencyOk) { // disallow transparent pixels in opaque regions of the input
            float eOutput_alpha = PYRAMID_FETCH_LAYER(gExemplarOutput, test.x, test.y, test.layer).w;

            // avoid copying partially transparent pixels into opaque regions of the input image
            if ( (eOutput_alpha < ALPHA_THRESHOLD) && (image_alpha > ALPHA_THRESHOLD ))
                return 10e30;
        }

        return calculateResidual(
                    offsets,
                    canvasOutputCache,
                    histogram,
                    advectedF, advectedB,
                    x, y, level, test);
    }
    return 10e30;
}

__global__ void scatter_kernel_keys(
        int originX, int originY,
        CudaImageBufferDevice<PatchXF> offsets,
        CudaImageBufferDevice<PatchXF> offsetsWorking,
        CudaImageBufferDevice<float> residualCache,
        CudaImageBufferDevice<Color4> canvasOutputCache,
        CudaImageBufferDevice<float> histogram,
        AdvectedBuffersDevice advectedF,
        AdvectedBuffersDevice advectedB,
        int sampleCount, int level, int exemplarWidth,
        int exemplarHeight, int keyIndex){

    int x = blockIdx.x * blockDim.x + threadIdx.x + originX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + originY;

    if (!offsets.inBounds(x,y,level) ||
            isImagePixelCulled(x, y)) {
        return;
    }

    PatchXF offset = offsets.pixel(x,y);

    Color4 keymask = (keyIndex < 0) ? Color4(-1.f) : toColor4(PYRAMID_FETCH_LAYER(gExemplarOutput, x, y, keyIndex));
    if(keymask.a > 0){
        //keymask, keep current offset
        offsetsWorking.setPixel(offset, x, y);
        return;
    }

    float cachedResidual = residualCache.pixel(x,y);
    float bestResidual = cachedResidual + 10e-7; // i.e, + epsilon;

    PatchXF oWorking = offset;

    // maybe enclose this in an if so we don't do a rando texlookup if we don't have to
    float image_alpha = PYRAMID_FETCH_LAYER(gImageBase, x, y, gCurrentBuffer).w;

    // used for generating the random offsets
    float dist = max(exemplarWidth, exemplarHeight);
    float closestDistance = 10.f;
    float scatterFactor = 0.9f;

    for(int i=0; (i < sampleCount / (gParams.lastKeyIndex - gParams.firstKeyIndex + 1)) && (dist * scatterFactor >= closestDistance); i++){

        float r = randFloat();
        float theta = randFloat() * M_2_PI;

        // scale vector so that its length is between dmin and dmax
        float dmax = dist;
        float dmin = dist * scatterFactor;

        float ex = oWorking.x + cos(theta) * (r * (dmax-dmin) + dmin);
        float ey = oWorking.y + sin(theta) * (r * (dmax-dmin) + dmin);

        clampToExemplar(&ex,&ey,exemplarWidth,exemplarHeight);

        int elayer = oWorking.layer;
        float eangle = screenToExemplarAngle(x, y, ex, ey, elayer);

        PatchXF oTest(ex,ey,eangle,0.f,elayer);

        float residual = test_offset(offsets, canvasOutputCache, histogram,
                                     advectedF, advectedB,
                                     x, y, oTest, level, image_alpha);
        if(residual < bestResidual){
            oWorking = oTest;
            bestResidual = residual;
        }

        RibbonP motion;
        float initialEx = ex;
        float initialEy = ey;

        for(int j=elayer-1; j>=gParams.firstKeyIndex; j--){
            motion = toRibbonP(PYRAMID_FETCH_LAYER(gCumulFieldB,ex,ey,j+1));
            if(!motion.isValid())
                break;
            ex += motion.x;
            ey += motion.y;
            eangle = screenToExemplarAngle(x, y, ex, ey, j);
            oTest = PatchXF(ex,ey,eangle,0.f,j);
            residual = test_offset(offsets, canvasOutputCache, histogram,
                                   advectedF, advectedB,
                                   x, y, oTest, level, image_alpha);
            if(residual < bestResidual){
                oWorking = oTest;
                bestResidual = residual;
            }
        }

        ex = initialEx;
        ey = initialEy;

        for(int j=elayer+1; j<=gParams.lastKeyIndex; j++){
            motion = toRibbonP(PYRAMID_FETCH_LAYER(gCumulFieldF,ex,ey,j-1));
            if(!motion.isValid())
                break;
            ex += motion.x;
            ey += motion.y;
            eangle = screenToExemplarAngle(x, y, ex, ey, j);
            oTest = PatchXF(ex,ey,eangle,0.f,j);
            residual = test_offset(offsets, canvasOutputCache, histogram,
                                   advectedF, advectedB,
                                   x, y, oTest, level, image_alpha);
            if(residual < bestResidual){
                oWorking = oTest;
                bestResidual = residual;
            }
        }

        dist *= scatterFactor;
    }

    if (bestResidual < cachedResidual) {
        offsetsWorking.setPixel(oWorking, x, y);
    }
}

__device__ PatchXF averageOffset(CudaImageBufferDevice<PatchXF> offsets, int x, int y, int level, int layer)
{
    PatchXF avg = PatchXF(0,0,0,1,layer);
    int sum = 0;
    for(int i = -1; i <= 1; i++){
	for(int j = -1; j <= 1; j++){
	    if(!offsets.inBounds(x+i, y+j, level) || (i==0 && j==0)){
		continue;
	    }
	    PatchXF n = offsets.pixel(x+i, y+j);
	    if(n.layer != layer){
		continue;
	    }
	    avg.x += n.x;
	    avg.y += n.y;
	    sum++;
	}
    }
    avg.x /= sum;
    avg.y /= sum;
    return avg;
}


__global__ void scatter_kernel(
        int originX, int originY,
        CudaImageBufferDevice<PatchXF> offsets,
        CudaImageBufferDevice<PatchXF> offsetsWorking,
        CudaImageBufferDevice<float> residualCache,
        CudaImageBufferDevice<Color4> canvasOutputCache,
        CudaImageBufferDevice<float> histogram,
        AdvectedBuffersDevice advectedF,
        AdvectedBuffersDevice advectedB,
        int sampleCount, int level, int exemplarWidth,
        int exemplarHeight, int keyIndex){

    int x = blockIdx.x * blockDim.x + threadIdx.x + originX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + originY;

    if (!offsets.inBounds(x,y,level) ||
            isImagePixelCulled(x, y)) {
        return;
    }

    PatchXF offset = offsets.pixel(x,y);

    Color4 keymask = (keyIndex < 0) ? Color4(-1.f) : toColor4(PYRAMID_FETCH_LAYER(gExemplarOutput, x, y, keyIndex));
    if(keymask.a > 0){
        //keymask, keep current offset
        offsetsWorking.setPixel(offset, x, y);
        return;
    }

    float cachedResidual = residualCache.pixel(x,y);
    float bestResidual = cachedResidual + 10e-7; // i.e, + epsilon;

    PatchXF oWorking = offset;

    // maybe enclose this in an if so we don't do a rando texlookup if we don't have to
    float image_alpha = PYRAMID_FETCH_LAYER(gImageBase, x, y, gCurrentBuffer).w;

    // used for generating the random offsets
    float dist = max(exemplarWidth, exemplarHeight);
    float closestDistance = 10.f;
    float scatterFactor = 0.9f;

    float2 scale = PYRAMID_FETCH(gImageScale, x, y);

    for(int i=0; (i < sampleCount) && (dist * scatterFactor >= closestDistance); i++){

        float r = randFloat();
        float theta = randFloat() * M_2_PI;

        // scale vector so that its length is between dmin and dmax
        float dmax = dist;
        float dmin = dist * scatterFactor;

        float ex = oWorking.x + cos(theta) * (r * (dmax-dmin) + dmin);
        float ey = oWorking.y + sin(theta) * (r * (dmax-dmin) + dmin);

        clampToExemplar(&ex,&ey,exemplarWidth,exemplarHeight);

        float eangle = screenToExemplarAngle(x, y, ex, ey, 0);

    	PatchXF oTest(ex,ey,eangle,0.f,0, scale.x, scale.y);

        float residual = test_offset(offsets, canvasOutputCache, histogram,
                                     advectedF, advectedB,
                                     x, y, oTest, level, image_alpha);
        if(residual < bestResidual){
            oWorking = oTest;
            bestResidual = residual;
        }

        dist *= scatterFactor;
    }

    // Test average offset
    PatchXF avg = averageOffset(offsets,x,y,level,0);
    avg.theta = screenToExemplarAngle(x, y, avg.x, avg.y, 0);
    float residual = test_offset(offsets, canvasOutputCache, histogram,
                                 advectedF, advectedB,
                                 x, y, avg, level, image_alpha);
    if(residual < bestResidual){
        oWorking = avg;
        bestResidual = residual;
    }

    if (bestResidual < cachedResidual) {
        offsetsWorking.setPixel(oWorking, x, y);
    }
}

/////////////////////////////////////////////////////////////////////////

__device__ PatchXF calculatePropagateOffset(
        CudaImageBufferDevice<PatchXF> offsets,
        int tx, int ty,
        float x, float y,
	int exemplarWidth, int exemplarHeight)
{
    PatchXF testOffset = offsets.pixelWrap(tx, ty);
    int outLayer = testOffset.layer;

    float2 basis0, basis1;

    float2 scale = PYRAMID_FETCH(gImageScale, tx, ty);
    testOffset.scaleU = scale.x;
    testOffset.scaleV = scale.y;

    screenToExemplarBasis(testOffset.theta, gParams.residualWindowScaleU * testOffset.scaleU, gParams.residualWindowScaleV * testOffset.scaleV, &basis0, &basis1);

    float outEx, outEy;
    calculateNewExemplarCoord(x-float(tx),y-float(ty),testOffset.x,testOffset.y,basis0,basis1,&outEx,&outEy);

    clampToExemplar(&outEx, &outEy, exemplarWidth, exemplarHeight);

    float outAngle = screenToExemplarAngle(x, y, outEx, outEy, outLayer);
    scale = PYRAMID_FETCH(gImageScale, x, y);

    return PatchXF(outEx,outEy,outAngle,0.f,outLayer,scale.x,scale.y);
}

__global__ void propagate_kernel_keys(
        int originX, int originY,
        CudaImageBufferDevice<PatchXF> offsets,
        CudaImageBufferDevice<PatchXF> offsetsWorking,
        CudaImageBufferDevice<float> residualCache,
        CudaImageBufferDevice<Color4> canvasOutputCache,
        CudaImageBufferDevice<float> histogram,
        AdvectedBuffersDevice advectedF,
        AdvectedBuffersDevice advectedB,
        int level, int keyWidth, int keyHeight,
        int jumpRange, int keyIndex ){

    int x = blockIdx.x * blockDim.x + threadIdx.x + originX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + originY;

    if (!offsets.inBounds(x,y,level)) {
        return;
    }

    if(isImagePixelCulled(x,y)) {
        PatchXF zero = PatchXF(0.0f, 0.0f, 0.0f, 0.0f, 0);
        offsetsWorking.setPixel(zero, x, y);
        return;
    }

    Color4 keymask = (keyIndex < 0) ? Color4(-1.f) : toColor4(PYRAMID_FETCH_LAYER(gExemplarOutput, x, y, keyIndex));
    if(keymask.a > 0){
        PatchXF oWorking = PatchXF(x, y, 0.f, 1.f, keyIndex);
        offsetsWorking.setPixel(oWorking, x, y);
        return;
    }

    PatchXF offset = offsets.pixel(x,y);

    float bestResidual = residualCache.pixel(x,y);

    // maybe enclose this in an if so we don't do a rando texlookup if we don't have to
    float image_alpha = PYRAMID_FETCH_LAYER(gImageBase, x, y, gCurrentBuffer).w;

    PatchXF oWorking = offset;

    for(int i=0; i<4; i++){
        int tx = 0;
        int ty = 0;

        if(i==0){
            tx = x+jumpRange;
            ty = y;
        }else if(i==1){
            tx = x-jumpRange;
            ty = y;
        }else if(i==2){
            tx = x;
            ty = y+jumpRange;
        }else if(i==3){
            tx = x;
            ty = y-jumpRange;
        }

	PatchXF oTest = calculatePropagateOffset(
		    offsets,
		    tx, ty, x, y,
		    keyWidth, keyHeight);

	float residual = test_offset(
                    offsets, canvasOutputCache, histogram,
                    advectedF, advectedB,
                    x, y, oTest, level, image_alpha);

        if(residual < bestResidual){
            oWorking = oTest;
            bestResidual = residual;
        }

	float ex = oTest.x;
	float ey = oTest.y;
	float eangle = oTest.theta;
	int elayer = oTest.layer;

        Color4 motion(-1.f);
        float initialEx = ex;
        float initialEy = ey;

        for(int j=elayer-1; j>=gParams.firstKeyIndex; j--){
            motion = toColor4(PYRAMID_FETCH_LAYER(gCumulFieldB,ex,ey,j+1));
            if(motion.r < -EPSILON)
                break;
            ex += motion.b;
            ey += motion.g;
            eangle = screenToExemplarAngle(x, y, ex, ey, j);
            oTest = PatchXF(ex,ey,eangle,0.f,j);
            residual = test_offset(
                        offsets, canvasOutputCache, histogram,
                        advectedF, advectedB,
                        x, y, oTest, level, image_alpha);

            if(residual < bestResidual){
                oWorking = oTest;
                bestResidual = residual;
            }
        }

        ex = initialEx;
        ey = initialEy;

        for(int j=elayer+1; j<=gParams.lastKeyIndex; j++){
            motion = toColor4(PYRAMID_FETCH_LAYER(gCumulFieldF,ex,ey,j-1));
            if(motion.r < -EPSILON)
                break;
            ex += motion.b;
            ey += motion.g;
            eangle = screenToExemplarAngle(x, y, ex, ey, j);
            oTest = PatchXF(ex,ey,eangle,0.f,j);
            residual = test_offset(
                        offsets, canvasOutputCache, histogram,
                        advectedF, advectedB,
                        x, y, oTest, level, image_alpha);

            if(residual < bestResidual){
                oWorking = oTest;
                bestResidual = residual;
            }
        }

    }
    offsetsWorking.setPixel(oWorking, x, y);
}

__global__ void propagate_kernel( 
        int originX, int originY,
        CudaImageBufferDevice<PatchXF> offsets,
        CudaImageBufferDevice<PatchXF> offsetsWorking,
        CudaImageBufferDevice<float> residualCache,
        CudaImageBufferDevice<Color4> canvasOutputCache,
        CudaImageBufferDevice<float> histogram,
        AdvectedBuffersDevice advectedF,
        AdvectedBuffersDevice advectedB,
        int level, int exemplarWidth, int exemplarHeight,
        int jumpRange, int keyIndex){

    int x = blockIdx.x * blockDim.x + threadIdx.x + originX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + originY;

    if (!offsets.inBounds(x,y,level)) {
        return;
    }

    if (isImagePixelCulled(x,y)) {
        PatchXF zero = PatchXF(0.0f, 0.0f, 0.0f, 0.0f, 0);
        offsetsWorking.setPixel(zero, x, y);
        return;
    }

    Color4 keymask = (keyIndex < 0) ? Color4(-1.f) : toColor4(PYRAMID_FETCH_LAYER(gExemplarOutput, x, y, keyIndex));
    if(keymask.a > 0){
        PatchXF oWorking = PatchXF(x, y, 0.f, 1.f, keyIndex);
        offsetsWorking.setPixel(oWorking, x, y);
        return;
    }

    float bestResidual = residualCache.pixel(x,y);

    // maybe enclose this in an if so we don't do a rando texlookup if we don't have to
    float image_alpha = PYRAMID_FETCH_LAYER(gImageBase, x, y, gCurrentBuffer).w;

    PatchXF oWorking = offsets.pixel(x,y);

    for(int i=0; i<4; i++){
        int tx = 0;
        int ty = 0;

        if(i==0){
            tx = x+jumpRange;
            ty = y;
        }else if(i==1){
            tx = x-jumpRange;
            ty = y;
        }else if(i==2){
            tx = x;
            ty = y+jumpRange;
        }else if(i==3){
            tx = x;
            ty = y-jumpRange;
        }

    	PatchXF oTest = calculatePropagateOffset(offsets,
		    tx, ty, x, y,
		    exemplarWidth, exemplarHeight);

        float residual = test_offset(
                    offsets, canvasOutputCache, histogram,
                    advectedF, advectedB,
                    x, y, oTest, level, image_alpha);

        if( residual < bestResidual ){
            oWorking = oTest;
            bestResidual = residual;
        }
    }

    PatchXF oTest = calculatePropagateOffset(
                        advectedF.offsets,
                        x, y, x, y,
                        exemplarWidth, exemplarHeight);

    if (oTest.hysteresis > 0) {
        float residual = test_offset(
                    offsets, canvasOutputCache, histogram,
                    advectedF, advectedB,
                    x, y, oTest, level, image_alpha);

        if( residual < bestResidual ){
            oWorking = oTest;
            bestResidual = residual;
        }
    }

    oTest = calculatePropagateOffset(
                        advectedB.offsets,
                        x, y, x, y,
                        exemplarWidth, exemplarHeight);

    if (oTest.hysteresis > 0) {
        float residual = test_offset(
                    offsets, canvasOutputCache, histogram,
                    advectedF, advectedB,
                    x, y, oTest, level, image_alpha);

        if( residual < bestResidual ){
            oWorking = oTest;
            bestResidual = residual;
        }
    }

    offsetsWorking.setPixel(oWorking, x, y);
}

void TexSynth::updateResidualCache(WorkingBuffers* buffers, int level)
{
    checkCUDAError("residualpre");

    dim3 numBlocks = buffers->offsets.blockCounts(level, kThreadsPerBlock);

    updateResidualCache_kernel<<<numBlocks,kThreadsPerBlock>>>(
        buffers->offsets,
        buffers->canvasOutputCache,
        buffers->offsetsHistogram,
        buffers->residualCache,
        buffers->advectedF,
        buffers->advectedB,
        level);

    cudaThreadSynchronize();
    checkCUDAError("residualpost");
}

void TexSynth::syncBuffers(WorkingBuffers* buffers, int level, int op_flags) {

    dim3 numBlocks = buffers->offsets.blockCounts(level, kThreadsPerBlock);

    if (!(op_flags & DONT_COPY_OFFSETS)) {
        buffers->offsets.copyWorkingToBase();
    }

    checkCUDAError("cachepre");

    bool calculateHist = false;
    for(int i=0; i<buffers->params().numStyles; ++i)
        calculateHist = calculateHist || (buffers->params().offsetsHistogramSlope[i] > 0.0f);

    if( calculateHist)
    {
        buffers->offsetsHistogram.reset();

        calculateOffsetsHistogram<float><<<numBlocks, kThreadsPerBlock>>>(level, buffers->offsets,
                                                                          buffers->offsetsHistogram);
        checkCUDAError("calculateOffsetsHistogram");

        cudaThreadSynchronize();

        // smooth the histogram
        buffers->offsetsHistogram.working().reset();

        convolutionRowsGPU(buffers->offsetsHistogram.working(),
                           buffers->offsetsHistogram.base(),
                           level, buffers->params().residualWindowSize);
        checkCUDAError("convolutionRowsGPU");

        convolutionColumnsGPU(buffers->offsetsHistogram.base(),
                              buffers->offsetsHistogram.working(),
                              level, buffers->params().residualWindowSize);
        checkCUDAError("convolutionColumnsGPU");

        cudaThreadSynchronize();
    }

    offsetsToOutput<<<numBlocks, kThreadsPerBlock>>>(level,
        buffers->offsets, buffers->canvasOutputCache);

    ////////
    /*offsetsToInput<<<numBlocks, kThreadsPerBlock>>>(level,
        buffers->offsets, buffers->canvasInputCache); */
    //////

    offsetsToDistTrans<<<numBlocks, kThreadsPerBlock>>>(level,
        buffers->offsets, buffers->distTransCache);

    cacheVelocities<<<numBlocks, kThreadsPerBlock>>>(level,
        buffers->canvasVelFCache, buffers->canvasVelBCache);

    cacheRibbon<<<numBlocks, kThreadsPerBlock>>>(level,
        buffers->ribbonField.base(), buffers->ribbonField.working());

    cudaThreadSynchronize();
    checkCUDAError("canvasOutputCachePost");

    if (!(op_flags & DONT_UPDATE_RESIDUAL)) {
        updateResidualCache(buffers, level);
    }
}


void TexSynth::propagate(WorkingBuffers* buffers, int level, int jumpRange, int keyIndex){
    int exemplarWidth = gExemplarBase.width();
    int exemplarHeight = gExemplarBase.height();

    checkCUDAError("proppre");

    dim3 numTotalBlocks = buffers->offsets.blockCounts(level, kThreadsPerBlock);

#ifdef DEBUG_CULLING
    int culled_count;
    cudaMemcpyFromSymbol(&culled_count,"gCulledCount",sizeof(culled_count));
#endif

    // We have to break up the kernels into multiple launches or else they tend to get
    // killed by the cuda driver's timeout mechanism.
    for (int blocksOriginY = 0; blocksOriginY < numTotalBlocks.y; blocksOriginY += kMaxBlocksY) {
        for (int blocksOriginX = 0; blocksOriginX < numTotalBlocks.x; blocksOriginX += kMaxBlocksX) {

            int countX = min(numTotalBlocks.x - blocksOriginX, kMaxBlocksX);
            int countY = min(numTotalBlocks.y - blocksOriginY, kMaxBlocksY);

            dim3 numBlocksThisIter(countX,countY);

            if(buffers->params().interpolateKeyFrame){
                propagate_kernel_keys<<<numBlocksThisIter,kThreadsPerBlock>>>(
                    blocksOriginX*kBlockSizeX, blocksOriginY*kBlockSizeY,
                    buffers->offsets, buffers->offsets.working(),
                    buffers->residualCache, buffers->canvasOutputCache, buffers->offsetsHistogram,
                    buffers->advectedF, buffers->advectedB,
                    level, exemplarWidth, exemplarHeight, jumpRange, keyIndex);
            }else{
                propagate_kernel<<<numBlocksThisIter,kThreadsPerBlock>>>(
                    blocksOriginX*kBlockSizeX, blocksOriginY*kBlockSizeY,
                    buffers->offsets, buffers->offsets.working(),
                    buffers->residualCache, buffers->canvasOutputCache, buffers->offsetsHistogram,
                    buffers->advectedF, buffers->advectedB,
                    level, exemplarWidth, exemplarHeight,jumpRange, keyIndex);
            }
        }
    }

#ifdef DEBUG_CULLING
    int culled_after_count;
    cudaMemcpyFromSymbol(&culled_after_count,"gCulledCount",sizeof(culled_after_count));
    printf("Culled before: %d, after %d\n", culled_count, culled_after_count);
#endif

    cudaThreadSynchronize();
    checkCUDAError("proppost");

    syncBuffers(buffers, level);
}


void TexSynth::scatter(WorkingBuffers* buffers, int level, int keyIndex){

    int exemplarWidth = gExemplarBase.width();
    int exemplarHeight = gExemplarBase.height();

    int sampleCount = buffers->params().scatterSamplesCount;

    dim3 numTotalBlocks = buffers->offsets.blockCounts(level, kThreadsPerBlock);

    // We have to break up the kernels into multiple launches or else they tend to get
    // killed by the cuda driver's timeout mechanism.
    for (int blocksOriginY = 0; blocksOriginY < numTotalBlocks.y; blocksOriginY += kMaxBlocksY) {
        for (int blocksOriginX = 0; blocksOriginX < numTotalBlocks.x; blocksOriginX += kMaxBlocksX) {

            int countX = min(numTotalBlocks.x - blocksOriginX, kMaxBlocksX);
            int countY = min(numTotalBlocks.y - blocksOriginY, kMaxBlocksY);

            dim3 numBlocksThisIter(countX,countY);

            if(buffers->params().interpolateKeyFrame){
                scatter_kernel_keys<<<numBlocksThisIter, kThreadsPerBlock>>>(
                    blocksOriginX*kBlockSizeX, blocksOriginY*kBlockSizeY,
                    buffers->offsets, buffers->offsets.working(),
                    buffers->residualCache, buffers->canvasOutputCache, buffers->offsetsHistogram,
                    buffers->advectedF, buffers->advectedB,
                    sampleCount, level, exemplarWidth, exemplarHeight, keyIndex);
            }else{
                scatter_kernel<<<numBlocksThisIter, kThreadsPerBlock>>>(
                    blocksOriginX*kBlockSizeX, blocksOriginY*kBlockSizeY,
                    buffers->offsets, buffers->offsets.working(),
                    buffers->residualCache, buffers->canvasOutputCache, buffers->offsetsHistogram,
                    buffers->advectedF, buffers->advectedB,
                    sampleCount, level, exemplarWidth, exemplarHeight, keyIndex);
            }
        }
    }

    cudaThreadSynchronize();
    checkCUDAError("scatter");

    syncBuffers(buffers, level);
}

__device__ float2 offsetUpsampleNearest(PatchXF coarse_xf, float delta_x, float delta_y,
                                        int exemplarWidth, int exemplarHeight)
{
    float2 eBasis0, eBasis1;
    screenToExemplarBasis(coarse_xf.theta, 1, 1, &eBasis0, &eBasis1);

    float2 interp;
    calculateNewExemplarCoord(delta_x, delta_y, coarse_xf.xy(), eBasis0, eBasis1, &interp);
    clampToExemplar(&interp.x, &interp.y, exemplarWidth, exemplarHeight);

    float up_x = (interp.x*2.0) + 0.5;
    float up_y = (interp.y*2.0) + 0.5;

    return make_float2(up_x, up_y);
}

__global__ void upsampleNearestKernel(
    CudaImageBufferDevice<PatchXF> offsets, CudaImageBufferDevice<PatchXF> offsetsWorking,
    CudaImageBufferDevice<float> residual, CudaImageBufferDevice<float> residualWorking,
    int exemplarWidth, int exemplarHeight, int fine_level){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!offsets.inBounds(x,y,fine_level,1)) {
        return;
    }

    float coarse_x = (float)(x-0.5) / 2.0f;
    float coarse_y = (float)(y-0.5) / 2.0f;
    int rcx = (int)round(coarse_x);
    int rcy = (int)round(coarse_y);

    float residual_nearest = residual.pixel(rcx, rcy);
    PatchXF coarse_xf = offsets.pixel(rcx, rcy);

    float2 up_xy = offsetUpsampleNearest(coarse_xf, coarse_x - rcx, coarse_y - rcy, exemplarWidth, exemplarHeight);
    int up_layer = coarse_xf.layer;
    float rWorking = residual_nearest;

    float orientation = screenToExemplarAngle(x, y, up_xy.x, up_xy.y, up_layer);

    float2 scale = PYRAMID_FETCH(gImageScale, x, y);
    PatchXF oWorking = PatchXF(up_xy.x, up_xy.y, orientation, 0.0f, up_layer,scale.x,scale.y);
    oWorking.luminanceShift = Luminance(); // XXX fix this if we start using luminance shift again.

    offsetsWorking.setPixel(oWorking, x, y);
    residualWorking.setPixel(rWorking, x, y);
}

__global__ void upsampleBilinearKernel(
    CudaImageBufferDevice<PatchXF> offsets, CudaImageBufferDevice<PatchXF> offsetsWorking,
    CudaImageBufferDevice<float> residual, CudaImageBufferDevice<float> residualWorking,
    int exemplarWidth, int exemplarHeight, int fine_level){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!offsets.inBounds(x,y,fine_level,1)) {
        return;
    }

    float coarse_x = (float)(x-0.5) / 2.0f;
    float coarse_y = (float)(y-0.5) / 2.0f;

    int lcx = floor(coarse_x);
    int lcy = floor(coarse_y);

    PatchXF o00 = offsets.pixel(lcx, lcy);
    PatchXF o10 = offsets.pixel(lcx+1, lcy);
    PatchXF o01 = offsets.pixel(lcx, lcy+1);
    PatchXF o11 = offsets.pixel(lcx+1, lcy+1);

    float alpha = coarse_x - lcx;
    float beta = coarse_y - lcy;

    float2 interp = ((1-alpha) * (1-beta) * o00.xy()) +
                    ((alpha) * (1-beta) * o10.xy()) +
                    ((1-alpha) * (beta) * o01.xy()) +
                    ((alpha) * (beta) * o11.xy());
    float2 up_xy = (interp * 2.0) + 0.5;

    const float MAX_INTERP_DIST_2 = 4.0 * 4.0;
    if (length2(interp - o00.xy()) > MAX_INTERP_DIST_2 ||
        length2(interp - o10.xy()) > MAX_INTERP_DIST_2 ||
        length2(interp - o01.xy()) > MAX_INTERP_DIST_2 ||
        length2(interp - o11.xy()) > MAX_INTERP_DIST_2) {

        int rcx = (int)round(coarse_x);
        int rcy = (int)round(coarse_y);

        PatchXF coarse_xf = offsets.pixel(rcx, rcy);

        up_xy = offsetUpsampleNearest(coarse_xf, coarse_x - rcx, coarse_y - rcy, exemplarWidth, exemplarHeight);
    }

    int up_layer = o00.layer;
    float rWorking = residual.pixel(lcx,lcy);

    float orientation = screenToExemplarAngle(x, y, up_xy.x, up_xy.y, up_layer);

    float2 scale = PYRAMID_FETCH(gImageScale, x, y);
    PatchXF oWorking = PatchXF(up_xy.x, up_xy.y, orientation, 0.0f, up_layer,scale.x,scale.y);
    oWorking.luminanceShift = Luminance(); // XXX fix this if we start using luminance shift again.

    offsetsWorking.setPixel(oWorking, x, y);
    residualWorking.setPixel(rWorking, x, y);
}


void TexSynth::upsample(WorkingBuffers* buffers, int fine_level)
{
    checkCUDAError("upsamplepre");

    dim3 numBlocks = buffers->offsets.blockCounts(fine_level, kThreadsPerBlock);
    int exemplarWidth = gExemplarBase.width();
    int exemplarHeight = gExemplarBase.height();

    upsampleBilinearKernel<<<numBlocks, kThreadsPerBlock>>>(
        buffers->offsets, buffers->offsets.working(),
        buffers->residualCache, buffers->residualCache.working(),
        exemplarWidth, exemplarHeight, fine_level);

    cudaThreadSynchronize();

    buffers->offsets.copyWorkingToBase();
    buffers->residualCache.copyWorkingToBase();

    checkCUDAError("upsamplepost");
}

void TexSynth::flipAnimBuffers()
{
    _currentBuffer = (_currentBuffer +1) % 2;
    cudaMemcpyToSymbol(gCurrentBuffer, &_currentBuffer,sizeof(int));
}

template<class T>
__device__ inline bool calculateTranslatedPosition(bool forwards,
                                                   CudaImageBufferDevice<T> src,
                                                   int x, int y, float &tx, float &ty,
                                                   int level)
{
    // Find the difference in position between pixel in frame i and i-1 (or i+1 if backwards)
    float2 forward, back;

    if(forwards){
        // If we are moving forward in time, use the backwards velocity to find where
        // we have come from.
        back = PYRAMID_FETCH_LAYER(gImageVelB, x, y, gCurrentBuffer);
        tx = (float)x + back.x;
        ty = (float)y + back.y;
        forward = PYRAMID_FETCH_LAYER(gImageVelF, tx, ty, (gCurrentBuffer+1)%2);
    }else{
        forward = PYRAMID_FETCH_LAYER(gImageVelF, x, y, gCurrentBuffer);
        tx = (float)x + forward.x;
        ty = (float)y + forward.y;
        back = PYRAMID_FETCH_LAYER(gImageVelB, tx, ty, (gCurrentBuffer+1)%2);
    }

    // The forward and backward directions shouldn't be more
    // different than a pixel across, but we allow some slop.
    const float valid_dist = 1.414 * 2;

    // Flip one vector to align them before comparing:
    forward.x = -forward.x; forward.y = -forward.y;

    float dist = sqrt((forward.x - back.x)*(forward.x - back.x) +
                      (forward.y - back.y)*(forward.y - back.y));

    return (src.inBounds(ceilf(tx),ceilf(ty),level) && src.inBounds(floorf(tx),floorf(ty),level)
           && (dist < valid_dist));
}

__global__ void advectKernel(
        bool forwards,
        CudaImageBufferDevice<PatchXF> src,
        AdvectedBuffersDevice dest,
        int level, int time_step, int op_flags,
        int exemplarWidth, int exemplarHeight)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dest.offsets.inBounds(x,y,level)){
        return;
    }

    if(isImagePixelCulled(x,y)){
        dest.base.setPixel(Color4(0), x, y);
        dest.output.setPixel(Color4(0), x, y);
        dest.offsets.setPixel(PatchXF(), x, y);
        return;
    }

    float4 frac_rand;
    float4 random = randomFloat4(x, y,frac_rand);

    float tx,ty;
    PatchXF output;

    // Check forward/backward to see if we have a valid advection position.
    if(!(op_flags & TexSynth::ADVECT_DO_NOT_CHECK_PREVIOUS) &&
            calculateTranslatedPosition<PatchXF>(forwards, src, x, y, tx, ty, level) &&
            !isImagePixelCulledPreviousFrame(tx,ty,level, 8.0))
    {
        int rtx = (int)round(tx);
        int rty = (int)round(ty);

        PatchXF nearest_xf = src.pixel(rtx, rty);

        float2 eBasis0, eBasis1;
        screenToExemplarBasis(nearest_xf.theta, 1, 1, &eBasis0, &eBasis1);

        float2 interp;
        calculateNewExemplarCoord(tx-rtx,ty-rty, nearest_xf.xy(), eBasis0, eBasis1, &interp);
        clampToExemplar(&interp.x, &interp.y, exemplarWidth, exemplarHeight);

        Color4 color      = toColor4(PYRAMID_FETCH_LAYER(gImageBase, x, y, gCurrentBuffer));
        Color4 prev_color = toColor4(PYRAMID_FETCH_LAYER(gImageBase, tx, ty, (gCurrentBuffer+1)%2));

        output.x = interp.x;
        output.y = interp.y;
        output.theta = nearest_xf.theta;
        output.luminanceShift = gParams.lumShiftEnable ? nearest_xf.luminanceShift + (color - prev_color).toLuminance() : Luminance();
        output.layer = nearest_xf.layer;
        output.hysteresis = 1;
        output.luminanceShift.a = 0.f;

        dest.base.setPixel(prev_color, x, y);
        dest.output.setPixel(toColor4(PYRAMID_FETCH_LAYER(gExemplarOutput, output.x, output.y, output.layer)), x, y);
        dest.timeStep.setPixel(time_step, x, y);
    }

    // If we are doing long range advection, check for a keyframe in between the
    // current frame and advection target. Also, if we picked a random value, check
    // if we could get a better one from the keyframe.
    if (op_flags & TexSynth::ADVECT_CHECK_FRAME_TO_KEY &&
            (time_step > 1 || output.hysteresis == 0)) {
        RibbonP frame_to_key;
        if (forwards) {
            frame_to_key = toRibbonP(PYRAMID_FETCH(gFrameToKeyRibbonB,x,y));
        } else {
            frame_to_key = toRibbonP(PYRAMID_FETCH(gFrameToKeyRibbonF,x,y));
        }
        if (frame_to_key.isValid() &&
                (abs(frame_to_key.time_step) < time_step || output.hysteresis == 0))
        {
            output.x = frame_to_key.x + x;
            output.y = frame_to_key.y + y;
            output.layer = frame_to_key.layer;
            output.hysteresis = 1.0f;
            output.luminanceShift = Luminance();
        }

        dest.base.setPixel(toColor4(PYRAMID_FETCH_LAYER(gExemplarBase, output.x, output.y, output.layer)), x, y);
        dest.output.setPixel(toColor4(PYRAMID_FETCH_LAYER(gExemplarOutput, output.x, output.y, output.layer)), x, y);
        dest.timeStep.setPixel(abs(frame_to_key.time_step), x, y);
    }

    // Border dithering is currently disabled
    // If dither, make this advection invalid.
    /*bool dither = advectionBorderDither(dest.offsets, tx, ty, level, random.y);
    if (dither || output.hysteresis == 0) {*/
    if (output.hysteresis == 0) {
        // If advection is invalid, just grab a random value.
        output = toPatchXF(random);

        // Hysteresis is set to zero for the random values so that
        // they will likely get replaced.
        output.hysteresis = 0.0f;

        output.luminanceShift = Luminance();

        dest.base.setPixel(Color4(0), x, y);
        dest.output.setPixel(Color4(0), x, y);
        dest.timeStep.setPixel(-1, x, y);
    }

    // Advection jitter (usually zero)
    output.x = output.x * (1 - gParams.advectionJitter) + frac_rand.x * (gParams.advectionJitter);
    output.y = output.y * (1 - gParams.advectionJitter) + frac_rand.y * (gParams.advectionJitter);

    // Reset the orientation to the difference between the image and exemplar
    // orientations. This used to be in "fixOffsets," but I'm not sure it's
    // actually the right thing to do. -fcole aug 25 2011
    output.theta = screenToExemplarAngle(x, y, output.x, output.y, output.layer);

    dest.offsets.setPixel(output, x, y);
}

void TexSynth::advectOffsets(WorkingBuffers* buffers, int level, int time_step, bool forwards, int op_flags)
{
    dim3 numBlocks = buffers->offsets.blockCounts(level, kThreadsPerBlock);

    checkCUDAError("advectpre");

    AdvectedBuffersHost* dest;
    if (forwards) {
        dest = &(buffers->advectedF);
    } else {
        dest = &(buffers->advectedB);
    }
    int exemplarWidth = gExemplarBase.width();
    int exemplarHeight = gExemplarBase.height();

    advectKernel<<<numBlocks, kThreadsPerBlock>>>(
        forwards, buffers->offsets, *dest, level,
        time_step, op_flags, exemplarWidth, exemplarHeight);

    cudaThreadSynchronize();
    checkCUDAError("advectpost");

    dest->offsets().copyTo(buffers->offsets.base());

    syncBuffers(buffers, level, DONT_COPY_OFFSETS | DONT_UPDATE_RESIDUAL);
    if(op_flags & ADVECT_UPDATE_RESIDUAL)
        updateResidualCache(buffers,level);

    // Copy the result to a side buffer for merging with
    // upsampled offsets later on.
    buffers->residualCache.base().copyTo(dest->residual());
}

__global__ void accumulateAdvectionKernel(
        bool forwards,
        CudaImageBufferDevice<RibbonP> dest,
        CudaImageBufferDevice<RibbonP> src,
        int level, int time_step, bool overwrite)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dest.inBounds(x,y,level)){
        return;
    }

    float tx,ty;

    RibbonP output; // Is initialized to invalid.

    Color4 inputcolor = toColor4(PYRAMID_FETCH_LAYER(gImageBase, x, y, gCurrentBuffer));
    bool masked_out = inputcolor.a == 0;

    // If we are outside the mask or if advection fails, just exit.
    if(masked_out ||
       !calculateTranslatedPosition<RibbonP>(forwards, src, x, y, tx, ty, level)) {
        if (overwrite) {
            dest.setPixel(output, x, y);
        }
        return;
    }

    int ftx = (int)floorf(tx);
    int fty = (int)floorf(ty);
    int ctx = (int)ceilf(tx);
    int cty = (int)ceilf(ty);

    RibbonP s_00 = src.pixel(ftx, fty);
    RibbonP s_10 = src.pixel(ctx, fty);
    RibbonP s_01 = src.pixel(ftx, cty);
    RibbonP s_11 = src.pixel(ctx, cty);
    bool any_valid = s_00.isValid() || s_10.isValid() || s_01.isValid() || s_11.isValid();
    if (!any_valid) {
        // There is nothing to advect, so exit.
        if (overwrite) {
            dest.setPixel(output, x, y);
        }
        return;
    }

    bool all_valid = s_00.isValid() && s_10.isValid() && s_01.isValid() && s_11.isValid();
    bool all_same_layer = (s_00.layer == s_10.layer == s_01.layer == s_11.layer);
    const float velBilerpThresh = 5.f;
    bool diverges_too_much = abs(s_00.x - s_10.x) > velBilerpThresh ||
            abs(s_00.x - s_01.x) > velBilerpThresh ||
            abs(s_00.x - s_11.x) > velBilerpThresh ||

            abs(s_00.y - s_10.y) > velBilerpThresh ||
            abs(s_00.y - s_01.y) > velBilerpThresh ||
            abs(s_00.y - s_11.y) > velBilerpThresh;

    if(!all_valid || !all_same_layer || diverges_too_much) {
        // bilinear interpolation is not valid either because
        // some of the neighboring velocities are invalid or
        // the velocity field diverges from more than 5px, so
        // use the nearest instead of interpolating.
        int rtx = (int)round(tx);
        int rty = (int)round(ty);

        if(ftx == rtx && fty == rty){
            output = s_00;
        }else if(ftx+1 == rtx && fty == rty){
            output = s_10;
        }else if(ftx == rtx && fty+1 == rty){
            output = s_01;
        }else{
            output = s_11;
        }
    } else {
        // Otherwise, bilinearly interpolate the 4 velocities
        float wx = tx - (float)ftx;
        float wy = ty - (float)fty;

        float rx0 = s_00.x * (1.f-wx) + s_10.x * wx;
        float ry0 = s_00.y * (1.f-wx) + s_10.y * wx;

        float rx1 = s_01.x * (1.f-wx) + s_11.x * wx;
        float ry1 = s_01.y * (1.f-wx) + s_11.y * wx;

        output.x = rx0 * (1.f-wy) + rx1 * wy;
        output.y = ry0 * (1.f-wy) + ry1 * wy;
        output.time_step = s_00.time_step;
        output.layer = s_00.layer;
    }

    output.x += (tx-x);
    output.y += (ty-y);
    output.time_step += time_step;

    RibbonP current = dest.pixel(x,y);
    if (overwrite || !current.isValid() || fabsf(current.time_step) >= fabsf(output.time_step)) {
        dest.setPixel(output,x,y);
    }
}

void TexSynth::accumulateRibbonField(WorkingBuffers* buffers, int level,
                                     bool forwards, int time_step, bool overwrite)
{
    dim3 numBlocks = buffers->canvasOutputCache.blockCounts(level, kThreadsPerBlock);

    checkCUDAError("advectAccumulatedPre");

    accumulateAdvectionKernel<<<numBlocks, kThreadsPerBlock>>>(
        forwards,
        buffers->ribbonField.working(),
        buffers->ribbonField,
        level, time_step, overwrite);

    cudaThreadSynchronize();
    checkCUDAError("advectAccumulatedPost");

    buffers->ribbonField.copyWorkingToBase();

    cumulativeAdvectionFieldToOutput<<<numBlocks, kThreadsPerBlock>>>(
        level,buffers->ribbonField,buffers->advectedF.output(),buffers->canvasOutputCache);

    cacheVelocities<<<numBlocks, kThreadsPerBlock>>>(level,
        buffers->canvasVelFCache, buffers->canvasVelBCache,
        forwards, !forwards);
}

__global__ void mergeUsingResidualKernel(
        int level,
        CudaImageBufferDevice<PatchXF> advectedOffsets,
        CudaImageBufferDevice<PatchXF> upsampledOffsets,
        CudaImageBufferDevice<PatchXF> resultOffsets,
        CudaImageBufferDevice<float> advectedResidual,
        CudaImageBufferDevice<float> upsampledResidual){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(!resultOffsets.inBounds(x,y,level)) {
        return;
    }

    PatchXF a = advectedOffsets.pixel(x,y);
    PatchXF u = upsampledOffsets.pixel(x,y);
    PatchXF r;

    float res = upsampledResidual.pixel(x,y);
    float resa = advectedResidual.pixel(x,y);

    if( resa < res){
        // this is the advected offsets
        r = a;
        r.hysteresis = 1.f;
    }else{
        // this is the offsets from the prev pass
        r = u;
        r.hysteresis = 0.f;
    }

    resultOffsets.setPixel(r,x,y);
}

__global__ void mergeRandomlyKernel(
        int level, bool forwards, float weightAdvected,
        CudaImageBufferDevice<PatchXF> advectedOffsets,
        CudaImageBufferDevice<PatchXF> upsampledOffsets,
        CudaImageBufferDevice<PatchXF> resultOffsets,
        bool force_valid){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(!resultOffsets.inBounds(x,y,level)) {
        return;
    }

    PatchXF a = advectedOffsets.pixel(x,y);
    PatchXF u = upsampledOffsets.pixel(x,y);
    PatchXF result;
    bool advected_valid = a.hysteresis > 0;

    RibbonP advected_ribbon, upsampled_ribbon;
    float4 random;
    randomFloat4(x, y, random);
    float r;
    if (forwards) {
        advected_ribbon = toRibbonP(PYRAMID_FETCH(gFrameToKeyRibbonB, x, y));
        upsampled_ribbon = toRibbonP(PYRAMID_FETCH(gFrameToKeyRibbonF, x, y));
        // Just to mix things up, use the x value on the forward pass and the y value
        // on the backwards pass.
        r = random.x;
    } else {
        advected_ribbon = toRibbonP(PYRAMID_FETCH(gFrameToKeyRibbonF, x, y));
        upsampled_ribbon = toRibbonP(PYRAMID_FETCH(gFrameToKeyRibbonB, x, y));
        r = random.y;
    }

    if( (advected_ribbon.isValid() && !upsampled_ribbon.isValid()) ||
            ((advected_valid || force_valid) && r < weightAdvected) ){
        // this is the advected offsets
        result = a;
    }else{
        // this is the offsets from the prev pass
        result = u;
    }

    resultOffsets.setPixel(result,x,y);
}

__global__ void mergeAdvectedKernel(
        int level, int fallback,
        CudaImageBufferDevice<PatchXF> advectedF,
        CudaImageBufferDevice<PatchXF> advectedB,
        CudaImageBufferDevice<PatchXF> lastPassOffsets,
        CudaImageBufferDevice<PatchXF> resultOffsets)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(!resultOffsets.inBounds(x,y,level)) {
        return;
    }

    PatchXF f = advectedF.pixel(x,y);
    PatchXF b = advectedB.pixel(x,y);
    PatchXF result;

    float4 random;
    randomFloat4(x, y, random);
    float r = random.x;

    if(f.hysteresis > 0 && b.hysteresis > 0) {
        result = (r < 0.5) ? f : b;
    } else if (f.hysteresis > 0) {
        result = f;
    } else if (b.hysteresis > 0) {
        result = b;
    } else {
        if (fallback == TexSynth::MERGE_FALLBACK_RANDOM) {
            result = toPatchXF(random);
        } else {
            result = lastPassOffsets.pixel(x,y);
        }
    }

    resultOffsets.setPixel(result,x,y);
}

void TexSynth::mergeUsingResidual(WorkingBuffers* buffers, int level, bool forwards)
{
    dim3 numBlocks = buffers->offsets.blockCounts(level, kThreadsPerBlock);

    cudaThreadSynchronize();
    checkCUDAError("mergepre");

    if(forwards) {
        mergeUsingResidualKernel<<<numBlocks, kThreadsPerBlock>>>(
            level,
            buffers->advectedF.offsets(), buffers->offsets, buffers->offsets.working(),
            buffers->advectedF.residual(), buffers->residualCache);
    } else {
        mergeUsingResidualKernel<<<numBlocks, kThreadsPerBlock>>>(
            level,
            buffers->advectedB.offsets(), buffers->offsets, buffers->offsets.working(),
            buffers->advectedB.residual(), buffers->residualCache);
    }

    cudaThreadSynchronize();
    checkCUDAError("mergepost");

    syncBuffers(buffers, level);
}

void TexSynth::mergeRandomly(WorkingBuffers* buffers, int level, bool forwards, float weightAdvected, bool force_valid)
{
    dim3 numBlocks = buffers->offsets.blockCounts(level, kThreadsPerBlock);

    cudaThreadSynchronize();
    checkCUDAError("mergepre");

    if(forwards) {
        mergeRandomlyKernel<<<numBlocks, kThreadsPerBlock>>>(
            level, forwards, weightAdvected,
            buffers->advectedF.offsets(), buffers->offsets, buffers->offsets.working(), force_valid);
    } else {
        mergeRandomlyKernel<<<numBlocks, kThreadsPerBlock>>>(
            level, forwards, weightAdvected,
            buffers->advectedB.offsets(), buffers->offsets, buffers->offsets.working(), force_valid);
    }

    cudaThreadSynchronize();
    checkCUDAError("mergepost");

    syncBuffers(buffers, level);
}

void TexSynth::mergeAdvectedBuffers(WorkingBuffers* buffers, int level, int merge_fallback)
{
    dim3 numBlocks = buffers->offsets.blockCounts(level, kThreadsPerBlock);

    cudaThreadSynchronize();
    checkCUDAError("mergepre");

    mergeAdvectedKernel<<<numBlocks, kThreadsPerBlock>>>(
                level, merge_fallback,
                buffers->advectedF.offsets(), buffers->advectedB.offsets(),
                buffers->offsets.base(), buffers->offsets.working());

    cudaThreadSynchronize();
    checkCUDAError("mergepost");

    syncBuffers(buffers, level);
}

__global__ void copyRandomKernel(CudaImageBufferDevice<PatchXF> dest, int level)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dest.inBounds(x,y,level)){
        return;
    }

    PatchXF output = toPatchXF(PYRAMID_FETCH(gRandom, x, y));
    output.luminanceShift = Luminance();

    dest.setPixel(output, x, y);

}

__global__ void copyFrameToKeyRibbonKernel(CudaImageBufferDevice<PatchXF> dest, int level, bool look_forwards)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dest.inBounds(x,y,level)){
        return;
    }

    RibbonP ribbon;
    if (look_forwards) {
        ribbon = toRibbonP(PYRAMID_FETCH(gFrameToKeyRibbonF, x, y));
    } else {
        ribbon = toRibbonP(PYRAMID_FETCH(gFrameToKeyRibbonB, x, y));
    }

    PatchXF output;

    if (ribbon.isValid()) {
        output.x = ribbon.x + x;
        output.y = ribbon.y + y;
        output.layer = ribbon.layer;
        output.hysteresis = 1.0f;
        output.luminanceShift = Luminance();

        // Reset the orientation to the difference between the image and exemplar
        // orientations. This used to be in "fixOffsets," but I'm not sure it's
        // actually the right thing to do. -fcole aug 25 2011
        output.theta = screenToExemplarAngle(x, y, output.x, output.y, output.layer);
    } else {
        output = toPatchXF(PYRAMID_FETCH(gRandom, x, y));
        output.luminanceShift = Luminance();
    }

    dest.setPixel(output, x, y);
}

// Copy random values from gRandom into offsetsWorking
void TexSynth::randomizeOffsets(WorkingBuffers* buffers, int level) {
    dim3 numBlocks = buffers->offsets.blockCounts(level, kThreadsPerBlock);

    checkCUDAError("randomizepre");

    copyRandomKernel<<<numBlocks, kThreadsPerBlock>>>(buffers->offsets.working(), level);

    syncBuffers(buffers, level);
    checkCUDAError("randomizepost");
}

void TexSynth::offsetsFromFrameToKeyRibbon(WorkingBuffers* buffers, int level, bool look_forwards)
{
    dim3 numBlocks = buffers->offsets.blockCounts(level, kThreadsPerBlock);

    copyFrameToKeyRibbonKernel<<<numBlocks, kThreadsPerBlock>>>(buffers->offsets.working(), level, look_forwards);

    syncBuffers(buffers, level);
}

void TexSynth::initialize_images(int width, int height)
{
    gImageBase.initialize(width,height,cudaFilterModeLinear,2);
    gImageDistanceTransform.initialize(width,height,cudaFilterModeLinear,2);
    gImageVelB.initialize(width,height,cudaFilterModeLinear,2);
    gImageVelF.initialize(width,height,cudaFilterModeLinear,2);
}

void TexSynth::initialize_exemplars(int width, int height, int numLayers)
{
    gExemplarBase.initialize(width,height,cudaFilterModeLinear,numLayers);
    gExemplarOutput.initialize(width,height,cudaFilterModeLinear,numLayers);
    gExemplarObjectIDs.initialize(width,height,cudaFilterModePoint,numLayers);
    gExemplarOrientation.initialize(width,height,cudaFilterModePoint,numLayers);
    gExemplarDistanceTransform.initialize(width,height,cudaFilterModeLinear,numLayers);
}

void TexSynth::initialize_cumulativeFields(int width, int height, int numLayers)
{
    gCumulFieldB.initialize(width,height,cudaFilterModeLinear,numLayers);
    gCumulFieldF.initialize(width,height,cudaFilterModeLinear,numLayers);
}

void TexSynth::uploadImage_float2(int level, TsImageLayer layer, const ImagePyramid<float2>& image)
{
    int width = image.width(level);
    int height = image.height(level);
    const void* storage = image.storagePointer(level);
    switch (layer)
    {
    case TS_LAYER_INPUT_SURF_ID :
        gImageSurfaceId.copyFromHost(width, height, cudaFilterModeLinear, storage);
        break;
    case TS_LAYER_INPUT_VEL_B :
        gImageVelB.copyFromHost(width, height, storage, _currentBuffer);
        break;
    case TS_LAYER_INPUT_VEL_F :
        gImageVelF.copyFromHost(width, height, storage, _currentBuffer);
        break;
    case TS_LAYER_INPUT_VEL_B_PREVIOUS :
        gImageVelB.copyFromHost(width, height, storage, (_currentBuffer+1)%2);
        break;
    case TS_LAYER_INPUT_VEL_F_PREVIOUS :
        gImageVelF.copyFromHost(width, height, storage, (_currentBuffer+1)%2);
        break;
    case TS_LAYER_INPUT_SCALE :
	gImageScale.copyFromHost(width, height, cudaFilterModeLinear, storage);
	break;
    default: assert(0); break;
    }
}

void TexSynth::uploadImage_Color4(int level, TsImageLayer layer, const ImagePyramid<Color4>& image, int styleLayer)
{
    int width = image.width(level);
    int height = image.height(level);
    const void* storage = image.storagePointer(level);
    switch (layer)
    {
    case TS_LAYER_EXEMPLAR_BASE :
        gExemplarBase.copyFromHost(width, height, storage, styleLayer);
        break;
    case TS_LAYER_EXEMPLAR_OUTPUT :
        gExemplarOutput.copyFromHost(width, height, storage, styleLayer);
        break;
    case TS_LAYER_INPUT_COLOR :
        gImageBase.copyFromHost(width, height, storage, _currentBuffer);
        break;
    case TS_LAYER_RANDOM :
        gRandom.copyFromHost(width, height, cudaFilterModePoint, storage);
        break;
    case TS_LAYER_GUIDE_F :
        gGuideF.copyFromHost(width, height, cudaFilterModeLinear, storage);
        break;
    case TS_LAYER_GUIDE_B :
        gGuideB.copyFromHost(width, height, cudaFilterModeLinear, storage);
        break;
    case TS_LAYER_CUMUL_FIELD_B :
        gCumulFieldB.copyFromHost(width, height, storage, styleLayer);
        break;
    case TS_LAYER_CUMUL_FIELD_F :
        gCumulFieldF.copyFromHost(width, height, storage, styleLayer);
        break;
    case TS_LAYER_FRAME_TO_KEY_RIBBON_B :
        gFrameToKeyRibbonB.copyFromHost(width, height, cudaFilterModeLinear, storage);
        break;
    case TS_LAYER_FRAME_TO_KEY_RIBBON_F :
        gFrameToKeyRibbonF.copyFromHost(width, height, cudaFilterModeLinear, storage);
        break;
    default: assert(0); break;
    }
}

void TexSynth::uploadImage_int(int level, TsImageLayer layer, const ImagePyramid<int>& image, int styleLayer)
{
    int width = image.width(level);
    int height = image.height(level);
    const void* storage = image.storagePointer(level);
    switch (layer)
    {
    case TS_LAYER_INPUT_ID :
        // Using nearest neighbor for IDs to avoid creating bogus interpolated values.
        gImageId.copyFromHost(width, height, cudaFilterModePoint, storage);
        break;
    case TS_LAYER_EXEMPLAR_OBJECT_IDS:
        gExemplarObjectIDs.copyFromHost(width, height, storage, styleLayer);
        break;
    default: assert(0); break;
    }
}

void TexSynth::uploadImage_float(int level, TsImageLayer layer, const ImagePyramid<float>& image, int styleLayer)
{
    int width = image.width(level);
    int height = image.height(level);
    const void* storage = image.storagePointer(level);
    switch (layer)
    {
    case TS_LAYER_EXEMPLAR_ORIENTATION :
        // Orientation textures use nearest neighbor sampling to avoid rotation interpolation problems.
        gExemplarOrientation.copyFromHost(width, height, storage, styleLayer);
        break;
    case TS_LAYER_EXEMPLAR_DIST_TRANS :
        gExemplarDistanceTransform.copyFromHost(width, height, storage, styleLayer);
        break;
    case TS_LAYER_INPUT_ORIENTATION :
        gImageOrientation.copyFromHost(width, height, cudaFilterModePoint, storage);
        break;
    case TS_LAYER_INPUT_DIST_TRANS :
        gImageDistanceTransform.copyFromHost(width, height, storage, _currentBuffer);
        break;
    default: assert(0); break;
    }
}

__global__ void copyExemplarDistTransK(CudaImageBufferDevice<Color4> dst, int layer)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,0))
        return;

    float dist = PYRAMID_FETCH_LAYER(gExemplarDistanceTransform, x, y, layer);
    Color4 out = Color4(dist / dst.width(), 0, 0, 1);
    dst.setPixel(out, x, y);
}

__global__ void copyInputDistTransK(CudaImageBufferDevice<Color4> dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,0))
        return;

    float dist = PYRAMID_FETCH_LAYER(gImageDistanceTransform, x, y, gCurrentBuffer);
    Color4 out = Color4(dist / dst.width(), 0, 0, 1);
    dst.setPixel(out, x, y);
}

__global__ void copyExemplarOrientationK(CudaImageBufferDevice<Color4> dst, int layer)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,0))
        return;

    const int halfsize = 10;

    float2 coord = make_float2(x,y);
    float theta = PYRAMID_FETCH_LAYER(gExemplarOrientation, x, y, layer);
    float2 dir = make_float2(cos(theta),sin(theta));

    float4 res, random;
    randomFloat4(x,y,res);
    float2 currentdir = dir;

    for(int i=1; i<=halfsize; i++) {
      coord  = coord + make_float2(currentdir.x,currentdir.y);
      randomFloat4(coord.x,coord.y,random);
      res   += random;

      theta = PYRAMID_FETCH_LAYER(gExemplarOrientation, coord.x, coord.y, layer);
      currentdir = make_float2(cos(theta),sin(theta));
    }

    coord = make_float2(x,y);
    currentdir = dir;
    for(int i=1; i<=halfsize; i++) {
        coord  = coord - make_float2(currentdir.x,currentdir.y);
        randomFloat4(coord.x,coord.y,random);
        res   += random;

        theta = PYRAMID_FETCH_LAYER(gExemplarOrientation, coord.x, coord.y, layer);
        currentdir = make_float2(cos(theta),sin(theta));
    }

    res = res/(2.0f*float(halfsize)+1.f);

    Color4 out = Color4(res.x,res.x,res.x,1.f);
    dst.setPixel(out, x, y);
}

__global__ void copyInputOrientationK(CudaImageBufferDevice<Color4> dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,0))
        return;

    const int halfsize = 10;

    float2 coord = make_float2(x,y);
    float theta = PYRAMID_FETCH(gImageOrientation, x, y);
    float2 dir = make_float2(cos(theta),sin(theta));

    float4 res, random;
    randomFloat4(x,y,res);
    float2 currentdir = dir;

    for(int i=1; i<=halfsize; i++) {
      coord  = coord + make_float2(currentdir.x,currentdir.y);
      randomFloat4(coord.x,coord.y,random);
      res   += random;

      theta = PYRAMID_FETCH(gImageOrientation, coord.x, coord.y);
      currentdir = make_float2(cos(theta),sin(theta));
    }

    coord = make_float2(x,y);
    currentdir = dir;
    for(int i=1; i<=halfsize; i++) {
        coord  = coord - make_float2(currentdir.x,currentdir.y);
        randomFloat4(coord.x,coord.y,random);
        res   += random;

        theta = PYRAMID_FETCH(gImageOrientation, coord.x, coord.y);
        currentdir = make_float2(cos(theta),sin(theta));
    }

    res = res/(2.0f*float(halfsize)+1.f);

    Color4 out = Color4(res.x,res.x,res.x,1.f);
    dst.setPixel(out, x, y);
}

__global__ void copyExemplarBaseK(CudaImageBufferDevice<Color4> dst, int layer)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,0))
        return;

    Color4 out = toColor4(PYRAMID_FETCH_LAYER(gExemplarBase, x, y, layer));
    dst.setPixel(out, x, y);
}

__global__ void copyExemplarOutputK(CudaImageBufferDevice<Color4> dst, int layer)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,0))
        return;

    Color4 out = toColor4(PYRAMID_FETCH_LAYER(gExemplarOutput, x, y, layer));
    dst.setPixel(out, x, y);
}

__global__ void copyInputBaseK(CudaImageBufferDevice<Color4> dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,0))
        return;

    Color4 out = toColor4(PYRAMID_FETCH_LAYER(gImageBase, x, y, gCurrentBuffer));
    dst.setPixel(out, x, y);
}

__global__ void copyInputScaleK(CudaImageBufferDevice<Color4> dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,0))
	return;

    float2 scale = PYRAMID_FETCH(gImageScale, x, y);
    Color4 out = Color4(scale.x/5.f, scale.y/5.f,0.f,1.f);
    dst.setPixel(out, x, y);
}

void TexSynth::copyInput(TsImageLayer layer, cudaArray* array, int width, int height, int styleLayer)
{
    CudaImageBufferHost<Color4> tempOutput;
    tempOutput.initialize(width, height);
    dim3 numBlocks = tempOutput.blockCounts(0,kThreadsPerBlock);

    switch (layer)
    {
    case TS_LAYER_INPUT_COLOR:
        copyInputBaseK<<<numBlocks, kThreadsPerBlock>>>(tempOutput);
        break;
    case TS_LAYER_INPUT_ORIENTATION:
        copyInputOrientationK<<<numBlocks, kThreadsPerBlock>>>(tempOutput);
        break;
    case TS_LAYER_INPUT_DIST_TRANS:
        copyInputDistTransK<<<numBlocks, kThreadsPerBlock>>>(tempOutput);
        break;
    case TS_LAYER_INPUT_SCALE:
	copyInputScaleK<<<numBlocks, kThreadsPerBlock>>>(tempOutput);
	break;
    case TS_LAYER_EXEMPLAR_BASE:
        copyExemplarBaseK<<<numBlocks, kThreadsPerBlock>>>(tempOutput,styleLayer);
        break;
    case TS_LAYER_EXEMPLAR_OUTPUT:
        copyExemplarOutputK<<<numBlocks, kThreadsPerBlock>>>(tempOutput,styleLayer);
        break;
    case TS_LAYER_EXEMPLAR_DIST_TRANS :
        copyExemplarDistTransK<<<numBlocks, kThreadsPerBlock>>>(tempOutput,styleLayer);
        break;
    case TS_LAYER_EXEMPLAR_ORIENTATION:
        copyExemplarOrientationK<<<numBlocks, kThreadsPerBlock>>>(tempOutput,styleLayer);
        break;
    default:
        assert(0); return;
    }
    tempOutput.copyTo(array,0);
}

#endif 
