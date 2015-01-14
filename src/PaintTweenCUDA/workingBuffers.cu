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

#include "workingBuffers.h"

#include "texSynth_interface.h"
#include "convolution.h"
#include "dataAccess.h"

#include <iostream>
#include <cstdio>

#include "cutil_math.h"

// the number of pixels in each threadblock. This parameter can affect performance,
// but the default of 16 is probably fine.
const int kBlockSizeX = 16;
const int kBlockSizeY = 16;
const dim3 kThreadsPerBlock(kBlockSizeX, kBlockSizeY);

__device__ Color4 HSVtoRGB(Color4 input_color)
{
    float h = input_color.r;
    float s = input_color.g;
    float v = input_color.b;
    float r,g,b;

    h  = h * 6; // Hue is assumed specified as 0-1, not 0-360.
    int hi = (int)h ;
    float f = h - hi ;
    float p = v*( 1 - s );
    float q = v*( 1 - f*s );
    float t = v*( 1 - ( 1 - f )*s );
    switch( hi )
    {
        case 0:
                r = v ; g = t ; b = p ;
                break ;
        case 1:
                r = q ; g = v ; b = p ;
                break ;
        case 2:
                r = p ; g = v ; b = t ;
                break ;
        case 3:
                r = p ; g = q ; b = v ;
                break ;
        case 4:
                r = t ; g = p ; b = v ;
                break ;
        case 5:
                r = v ; g = p ; b = q ;
                break ;

    }
    Color4 RGB_value;
    RGB_value.r = r;
    RGB_value.g = g;
    RGB_value.b = b;
    RGB_value.a = input_color.a;
    return RGB_value;
}

__global__ void copyOffsetK(CudaImageBufferDevice<Color4> dst, int level,
            CudaImageBufferDevice<PatchXF> src, int max_x_offset, int max_y_offset,
            int numStyles, int mode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,level))
        return;

    PatchXF offset = src.pixel(x,y);
    Color4 out;

    float scale_x = offset.x / (float)max_x_offset;
    float scale_y = (offset.y +  offset.layer * max_y_offset) / (float)(max_y_offset * numStyles);
    if (mode == 0) {
        if (scale_x < 0 || scale_x > 1 || scale_y < 0 || scale_y > 1) {
            // Bad offset value.
            Color4 bad = Color4(0,0,1,1);
            out = bad;
        } else {
            Color4 ok = Color4(scale_x,scale_y,0,1);
            out = ok;
        }
    } else if (mode == 1) {
        out = Color4(offset.hysteresis, 0, 0, 1);
    } else if (mode == 2) {
        float angle = offset.theta * M_2_PI;
        if(fabs(angle) > 1.f)
            out = Color4(1, 0, 0, 1);
        else
            out = Color4(0, -clamp(angle,-1.0f,0.f), clamp(angle,0.f,1.0f), 1);
    }else if(mode==3) {
	float2 diff = (src.pixel(x+1,y).xy() - offset.xy());
	float mag = diff.x*diff.x + diff.y*diff.y -1;
	out = Color4(-clamp(mag,-1.0f,0.f), clamp(mag,0.f,1.0f),0.f,1.f);
    }else if(mode==4) {
	float2 diff = fabs(src.pixel(x,y+1).xy() - offset.xy());
	float mag = diff.x*diff.x + diff.y*diff.y -1;
	out = Color4(-clamp(mag,-1.0f,0.f), clamp(mag,0.f,1.0f),0.f,1.f);
    }else if(mode==5) {
	out = Color4(offset.scaleU/5.0f,offset.scaleV/5.0f,0.f,1.f);
    }
    dst.setPixel(out, x, y);
}

__global__ void copyIdsK(CudaImageBufferDevice<Color4> dst, int level,
            CudaImageBufferDevice<PatchXF> src, int numStyles)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,level))
        return;

    PatchXF offset = src.pixel(x,y);
    Color4 out;

    out.r = (float)offset.layer / (float)numStyles;
    out.g = out.r;
    out.b = out.r;

    dst.setPixel(out, x, y);
}

__global__ void copyVelocityK(TsParameters params,
                              CudaImageBufferDevice<Color4> dst, int level,
                              CudaImageBufferDevice<float2> src, float gain)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (!dst.inBounds(x,y,level))
      return;

  float2 vel = src.pixel(x,y);

  float hue = (atan2(vel.y, vel.x) + 3.1415) / (2 * 3.1415);
  float value = min(length(vel) * gain * 0.1, 1.0);
  Color4 hsv = Color4(hue, 1.0, value, 1.0);
  Color4 out = HSVtoRGB(hsv);

  dst.setPixel(out, x, y);
}

__global__ void copyRibbonK(TsParameters params,
                              CudaImageBufferDevice<Color4> dst, int level,
                              CudaImageBufferDevice<RibbonP> src)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (!dst.inBounds(x,y,level))
      return;

  const float max_speed = 20;
  const float max_time_step = 10;

  RibbonP ribbon = src.pixel(x,y);
  float2 vel = make_float2(ribbon.x, ribbon.y);

  float hue = (atan2(vel.y, vel.x) + 3.1415) / (2 * 3.1415);
  float value = min(length(vel) / max_speed, 1.0);
  Color4 hsv = Color4(hue, value, (float)(fabsf(ribbon.time_step)) / max_time_step, 1.0);
  Color4 out = HSVtoRGB(hsv);
  if (!ribbon.isValid()) {
      out = Color4(1,0,0,1);
  }

  dst.setPixel(out, x, y);
}

__global__ void copyTimeStepK(CudaImageBufferDevice<Color4> dst,
                              CudaImageBufferDevice<int> src,
                              int level)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,level))
        return;

    int step = src.pixel(x,y);
    Color4 out;
    switch (step) {
        case 0: out = Color4(0,0,0,1); break;
        case 1: out = Color4(1,0,0,1); break;
        case 2: out = Color4(0,1,0,1); break;
        case 4: out = Color4(0,0,1,1); break;
        case 8: out = Color4(1,1,0,1); break;
        case 16: out = Color4(1,0,1,1); break;
        case 32: out = Color4(0,1,1,1); break;
        default: out = Color4(1,1,1,1); break;
    }

    dst.setPixel(out, x, y);
}

__global__ void copyCompareK(CudaImageBufferDevice<Color4> dst,
                              CudaImageBufferDevice<float2> src,
                              int level, float viz_scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,level))
        return;

    float2 s = src.pixel(x,y);
    Color4 out = Color4(s.x*viz_scale, s.y*viz_scale, 0, 1);
    dst.setPixel(out, x, y);
}

__global__ void copyResidualK(CudaImageBufferDevice<Color4> dst,
                              CudaImageBufferDevice<float> src,
                              int level, float viz_scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,level))
        return;

    float residual = src.pixel(x,y);
    float r,g,b;

    if (residual > 0) {
        residual = residual * viz_scale;
        residual = sqrtf(residual);

        b = min(residual / 0.33f, 1.0f);
        g = min((residual - 0.33f) / 0.33f, 1.0f);
        r = min((residual - 0.66f) / 0.33f, 1.0f);
    } else {
        residual = sqrtf(-residual);
        residual = residual * 3;
        b = 0;
        g = min(residual, 1.0f);
        r = 0;
    }

    Color4 o = Color4(r,g,b,1);
    dst.setPixel(o,x,y);
}

__device__ void colormap(float f, float & r, float & g, float &b)
{
        const float dx = 0.8;
        float v = (6.0 - 2.0*dx)*f + dx;
        r = fmax(0.0, (3.0 - fabs(v-4.0) - fabs(v-5.0))/2.0);
        g = fmax(0.0, (4.0 - fabs(v-2.0) - fabs(v-4.0))/2.0);
        b = fmax(0.0, (3.0 - fabs(v-1.0) - fabs(v-2.0))/2.0);
}

template <class T>
__global__ void copyOffsetsHistogramK(CudaImageBufferDevice<Color4> dst,
                              CudaImageBufferDevice<T> src,
                              int level, float maxValue, int styleIndex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,level))
        return;

    float bin = (float)src.pixel(x,y + styleIndex * dst.height(level)) / maxValue;
    float r,g,b;
    colormap(bin,r,g,b);

    Color4 o(r,g,b,1);
    dst.setPixel(o,x,y);
}

template <class T>
__global__ void copyOffsetsHistogram(CudaImageBufferDevice<T> dst,
                              CudaImageBufferDevice<T> src,
                              int level, int styleIndex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!dst.inBounds(x,y,level))
        return;

    float bin = (float)src.pixel(x,y + styleIndex * dst.height(level));
    dst.setPixel(bin,x,y);
}

void WorkingBuffers::copyOutputArray(TsOutputType outputType, int level, cudaArray* output, int styleIndex,
                                     float vizGain, bool vizNormalize, int vizMode)
{
    dim3 numBlocks = canvasOutputCache.blockCounts(level, kThreadsPerBlock);

    if (!isInitialized()) {
        printf("ERROR: attempting to read when cuda memory is not allocated.\n");
        return;
    }

    DataAccess& data_access = DataAccess::instance();
    int numStyles = data_access.getNumStyles();

    float maxValue = 1.0;
    bool calculateHist = false;
    int max_x, max_y;
    tempOutput.initialize(_image_width, _image_height);

    switch (outputType) {
    case TS_OUTPUT_CANVAS :
        canvasOutputCache.copyTo(output, level);
        break;
    case TS_OUTPUT_OFFSET :
        max_x = _style_width >> level;
        max_y = (_style_height / numStyles) >> level;
        copyOffsetK<<<numBlocks, kThreadsPerBlock>>>( tempOutput, level,
            offsets, max_x, max_y, numStyles, vizMode);

        cudaThreadSynchronize();

        tempOutput.copyTo(output, level);
        break;

    case TS_OUTPUT_VEL_F :
        if (vizNormalize) {
            maxValue = bufferMax(residualCache.base(), level, 0, 1);
        }
        copyVelocityK<<<numBlocks, kThreadsPerBlock>>>(_params,
            tempOutput, level, canvasVelFCache, vizGain / maxValue);

        cudaThreadSynchronize();

        tempOutput.copyTo(output, level);
        break;

    case TS_OUTPUT_VEL_B :
        if (vizNormalize) {
            maxValue = bufferMax(residualCache.base(), level, 0, 1);
        }
        copyVelocityK<<<numBlocks, kThreadsPerBlock>>>(_params,
            tempOutput, level, canvasVelBCache, vizGain / maxValue);

        cudaThreadSynchronize();

        tempOutput.copyTo(output, level);
        break;

    case TS_OUTPUT_RIBBON_F :
        copyRibbonK<<<numBlocks, kThreadsPerBlock>>>(_params,
            tempOutput, level, ribbonField.base());

        cudaThreadSynchronize();

        tempOutput.copyTo(output, level);
        break;

    case TS_OUTPUT_RIBBON_B :
        copyRibbonK<<<numBlocks, kThreadsPerBlock>>>(_params,
            tempOutput, level, ribbonField.working());

        cudaThreadSynchronize();

        tempOutput.copyTo(output, level);
        break;

    case TS_OUTPUT_RESIDUAL :
        // N.B.: this copy code seems to fail on images larger than perhaps 1000x700.
        // It works fine on the scene with Red but on the Rifle scene produces
        // weird stripes that look like unaligned memory accesses. Maybe a bug in
        // Cuda? -fcole aug 2 2011.

        if (vizNormalize) {
            maxValue = bufferMax(residualCache.base(), level, 0, 1);
        }
        //printf("Max residual %f\n", maxValue);
        copyResidualK<<<numBlocks, kThreadsPerBlock>>>(tempOutput,
            residualCache, level, vizGain / maxValue);

        cudaThreadSynchronize();

        tempOutput.copyTo(output, level);
        break;

    case TS_OUTPUT_DIST_TRANS :
        copyCompareK<<<numBlocks, kThreadsPerBlock>>>(tempOutput,
            distTransCache, level, 0.1);

        cudaThreadSynchronize();

        tempOutput.copyTo(output, level);
        break;

    case TS_OUTPUT_ADVECTED_F :
        if (vizMode == 0) {
            advectedF.output().copyTo(output, level);
        } else if (vizMode >= 1 && vizMode <= 3) {
            copyOffsetK<<<numBlocks, kThreadsPerBlock>>>( tempOutput, level,
                advectedF.offsets(), _style_width >> level, (_style_height / numStyles) >> level,
                numStyles, vizMode-1);

            cudaThreadSynchronize();

            tempOutput.copyTo(output, level);
        } else {
            copyTimeStepK<<<numBlocks, kThreadsPerBlock>>>(tempOutput,
                advectedF.timeStep(), level);

            cudaThreadSynchronize();

            tempOutput.copyTo(output, level);
        }

        break;

    case TS_OUTPUT_ADVECTED_B :
        if (vizMode == 0) {
            advectedB.output().copyTo(output, level);
        } else if (vizMode >= 1 && vizMode <= 3) {
            copyOffsetK<<<numBlocks, kThreadsPerBlock>>>( tempOutput, level,
                advectedB.offsets(), _style_width >> level, (_style_height / numStyles) >> level,
                numStyles, vizMode-1);

            cudaThreadSynchronize();

            tempOutput.copyTo(output, level);
        } else {
            copyTimeStepK<<<numBlocks, kThreadsPerBlock>>>(tempOutput,
                advectedB.timeStep(), level);

            cudaThreadSynchronize();

            tempOutput.copyTo(output, level);
        }
        break;

    case TS_OUTPUT_HISTOGRAM :

        cudaThreadSynchronize();

	for(int i=0; i<_params.numStyles; ++i)
	    calculateHist = calculateHist || (_params.offsetsHistogramSlope[i] > 0.0f);

	if(calculateHist)
        {
            maxValue = max(maxValue, bufferMax(offsetsHistogram.base(),level,styleIndex,_params.numStyles));
            cudaThreadSynchronize();
        }

        tempOutputHistogram.initialize(_style_width, _style_height/_params.numStyles);
        numBlocks = tempOutputHistogram.blockCounts(level, kThreadsPerBlock);

        copyOffsetsHistogramK<float><<<numBlocks, kThreadsPerBlock>>>(tempOutputHistogram,
            offsetsHistogram, level, maxValue, styleIndex);

        cudaThreadSynchronize();

        tempOutputHistogram.copyTo(output, level);

        break;

    case TS_OUTPUT_ID:
        copyIdsK<<<numBlocks, kThreadsPerBlock>>>(tempOutput,
            level, offsets, numStyles);

        cudaThreadSynchronize();

        tempOutput.copyTo(output, level);
        break;

    default:
        break;
    }

    checkCUDAError("copyOutputArray");
}


