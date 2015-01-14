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

#ifndef _TEXSYNTH_UTIL_
#define _TEXSYNTH_UTIL_

#include "types.h"
#include "cutil_math.h"

#include <cuda_runtime.h>
#include <cstdio>

inline __host__ __device__ float length2(const float2& vec)
{
    return dot(vec, vec);
}

__device__ void clampToExemplar(int& x, int& y, int width, int height){
    if(x<0) x = 0;
    if(y<0) y = 0;

    if(x >= width ) x = width-1;
    if(y >= height) y = height-1;
}

__device__ void clampToExemplar(float* x, float* y, int width, int height){

    if(*x<0) *x = 0;
    if(*y<0) *y = 0;

    if(*x >= width ) *x = width-1;
    if(*y >= height) *y = height-1;
}

__device__ float sumSquaredDiff(float4* a, float4* b){
    float xd = a->x - b->x;
    float yd = a->y - b->y;
    float zd = a->z - b->z;
    float ad = a->w - b->w;

    return xd*xd + yd*yd + zd*zd + ad*ad;
}

__device__ float sumSquaredDiff(float4* a, float4* b, float alpha){
    float xd = a->x - b->x;
    float yd = a->y - b->y;
    float zd = a->z - b->z;
    float ad = a->w - b->w;

    return xd*xd + yd*yd + zd*zd + alpha*ad*ad;
}

inline __device__ float sumSquaredDiff(Color4 a, Color4 b){
    float rd = a.r - b.r;
    float gd = a.g - b.g;
    float bd = a.b - b.b;
    float ad = a.a - b.a;

    return rd*rd + gd*gd + bd*bd + ad*ad;
}

inline __device__ float sumSquaredDiff(Color4 a, Color4 b, float alpha){
    float rd = a.r - b.r;
    float gd = a.g - b.g;
    float bd = a.b - b.b;
    float ad = a.a - b.a;

    return rd*rd + gd*gd + bd*bd + alpha*ad*ad;
}

__device__ void screenToExemplarBasis(float angle, float scale_u, float scale_v, float2* basis0, float2* basis1){
    basis0->x = cos(angle);
    basis0->y = sin(angle);

    basis1->x = -basis0->y * scale_v; 	// -sin(angle)
    basis1->y = basis0->x * scale_v; 	// cos(angle)

    basis0->x *= scale_u;
    basis0->y *= scale_u;
}

__device__ void calculateNewExemplarCoord(float i, float j, float ex, float ey, float2 eBasis0, float2 eBasis1, float* outEx, float* outEy){
    float x = i*eBasis0.x + j*eBasis1.x;
    float y = i*eBasis0.y + j*eBasis1.y;
    *outEx = ex+x;
    *outEy = ey+y;
}

__device__ void calculateNewExemplarCoord(float i, float j, float2 e, float2 eBasis0, float2 eBasis1, float2* out){
    float2 coord;
    coord.x = i*eBasis0.x + j*eBasis1.x;
    coord.y = i*eBasis0.y + j*eBasis1.y;
    *out = e + coord;
}

// clamp x to a value between a and b
__device__ Color4 clamp(Color4 x, float a, float b){
    return Color4(clamp(x.r,a,b),clamp(x.g,a,b),clamp(x.b,a,b),clamp(x.a,a,b));
}


__device__ bool inBounds(int x, int y, int imageWidth, int imageHeight, int border = 0) {
    return x >= border && y >= border && x < imageWidth - border && y < imageHeight - border;
}


__device__ float bilerp(float x00, float x01, float x10, float x11, float alpha, float beta) {
    return (x00 * (1-alpha) * (1-beta)) + (x01 * alpha * (1-beta)) +
            (x10 * (1-alpha) * beta) + (x11 * alpha * beta);
}


/* Quasi random number generator
 * combined tausworthe generator  - see nvidia forums topic 155695
 * and gpugems 3 ch 37 section 4
 */

__constant__ unsigned int shift1[4] = {6, 2, 13, 3};
__constant__ unsigned int shift2[4] = {13, 27, 21, 12};
__constant__ unsigned int shift3[4] = {18, 2, 7, 13};
__constant__ unsigned int offset[4] = {4294967294, 4294967288, 4294967280, 4294967168};

__shared__ unsigned int randStates[32];

__device__ unsigned int TausStep(unsigned int &z, int S1, int S2, int S3, unsigned int M)
{
	unsigned int b = (((z << S1) ^ z) >> S2);
	return z = (((z &M) << S3) ^ b);
}

__device__ unsigned int randFloat()
{
	TausStep (randStates[threadIdx.x & 31], shift1[threadIdx.x&3],
		   shift2[threadIdx.x&3],shift3[threadIdx.x&3],offset[threadIdx.x&3]);

	return (float) (randStates[(threadIdx.x)&31]^randStates[(threadIdx.x+1)&31]
			^randStates[(threadIdx.x+2)&31]^randStates[(threadIdx.x+3)&31])
			   / powf(2.f, 32.f);
}

__device__ float Csrgb(float Clinear){
  float a = 0.055;

  if(Clinear > 0.0031308)
    return (1.0+a) * powf(Clinear, (1.0/2.4)) - a;

  else
    return 12.92*Clinear;
}

__device__ float Clinear(float Csrgb){
  float a = 0.055;

  if(Csrgb > 0.04045)
    return pow( ((Csrgb + a) / (1.0 + a)), 2.4);

  else
    return Csrgb/12.92;
}

__device__ float f_Lab(float t){
  if (t > powf(6.0/29.0, 3))
    return powf(t, 1.0/3.0);

  else
    return (1.0/3.0) * powf((29.0/6.0) , 2) * t + (4.0/29.0);

}

__device__ float f_inv_Lab(float t){
  if ( t > (6.0/29.0))
    return powf(t, 3);

  else
    return 3.* pow((6./29.), 2) * (t - (4./29.));

}

inline __device__ Color4 toColor4(float4 c) {
    return Color4(c.x, c.y, c.z, c.w);
}

inline __device__ Color4 toColor4(float4 c, Luminance lum_shift) {
    return Color4(c.x+lum_shift.y, c.y+lum_shift.y, c.z+lum_shift.y, c.w+lum_shift.a);
}


inline __device__ PatchXF toPatchXF(float4 c) {
    return PatchXF(c.x, c.y, c.z, 0.f, c.w);
}

inline __device__ RibbonP toRibbonP(float4 c) {
    return RibbonP(c.z, c.y, c.x, c.w);
}

__device__ Color4 RGBtoLAB(Color4 input_color)
{
    // change each pixel to LAB
    // using D65 as a reference point:
    float refX = 95.047;
    float refY = 100.0;
    float refZ = 108.883;

    float  rLinear = 100*Clinear(input_color.r);
    float  gLinear = 100*Clinear(input_color.g);
    float  bLinear = 100*Clinear(input_color.b);
    float alpha = input_color.a;

    float x = 0.4124 * rLinear + 0.3576 * gLinear + 0.1805 * bLinear;
    float y = 0.2126 * rLinear + 0.7152 * gLinear + 0.0722 * bLinear;
    float z = 0.0193 * rLinear + 0.1192 * gLinear + 0.9505 * bLinear;

    // dividing by 100, 200 here approximately normalizes the LAB value to be 
    // in the same approximate range as the RGB [0,1] values. 
    float L = (116.f * f_Lab(y/refY) - 16)/100.f;
    float a = (500.f * ( f_Lab(x/refX) - f_Lab(y/refY) ) + 100) / 200.f;
    float b = (200.f * ( f_Lab(y/refY) - f_Lab(z/refZ) ) + 100) / 200.f;

    // note: alpha channel is unchanged.
    Color4 LAB_value = Color4(L, a, b, alpha);
    return LAB_value;
}

__device__ Color4 LABtoRGB(Color4 input_color)
{
   // change each pixel to RGB
   // using D65 as a reference point:
   float refX = 95.047;
   float refY = 100.0;
   float refZ = 108.883;

   float L = input_color.r;
   float a = input_color.g;
   float b = input_color.b;
   float alpha = input_color.a;

   // 1/500 = 0.002    1/200 = 0.005
   float x = refX * f_inv_Lab ((1./116.) * (L + 16) + (0.002 * a));
   float y = refY * f_inv_Lab ((1./116.) * (L + 16));
   float z = refZ * f_inv_Lab ((1./116.) * (L + 16) - (0.005 * b));

   float rLinear =   3.2406 * x - 1.5372 * y - 0.4986 * z;
   float gLinear = -0.9689 * x + 1.8758 * y + 0.0415 * z;
   float bLinear =  0.0557 * x  - 0.2040 * y + 1.0570 * z;

   // note: alpha channel is unchanged.
   Color4 RGB_value;
   RGB_value.r = Csrgb(rLinear);
   RGB_value.g = Csrgb(gLinear);
   RGB_value.b = Csrgb(bLinear);
   RGB_value.a = alpha;
   return RGB_value;
}

#endif
