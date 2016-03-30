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

#ifndef TYPES_H
#define TYPES_H

#include <cassert>
#include <vector>
#include <algorithm>
#include <math.h>

#include <cuda_runtime.h>
	
#define EPSILON 0.000001f
#define ALPHA_THRESHOLD 0.99f

// Color is treated as BGR because Shake stores its color arrays as BGR triples
struct Color
{
public:
    __host__ __device__ Color ():r(0.f),g(0.f),b(0.f){}
    __host__ __device__ Color (float rr, float gg, float bb):r(rr),g(gg),b(bb){}
    
    float r,g,b;
};

struct Luminance
{
public:
    __host__ __device__ Luminance ():y(0.f),a(0.f){}
    __host__ __device__ Luminance (float yy, float aa):y(yy),a(aa){}

    __host__ __device__ inline Luminance operator + (const Luminance& right) const { return Luminance(y + right.y, a + right.a); }

    __host__ __device__ inline Luminance operator - (const Luminance& right) const { return Luminance(y - right.y, a - right.a); }

    __host__ __device__ inline Luminance operator * (const float right) const { return Luminance(y * right, a * right); }

    float y,a;
};

struct Color4
{
public:
    __host__ __device__ Color4 ():r(0.f),g(0.f),b(0.f),a(1.f){}
    __host__ __device__ Color4 (float rr, float gg, float bb, float aa):r(rr),g(gg),b(bb),a(aa){}
    __host__ __device__ Color4 (float v):r(v),g(v),b(v),a(v){}

    __host__ __device__ inline bool operator == (const Color4& other) const { return fabs(r - other.r)<EPSILON && fabs(g - other.g)<EPSILON && fabs(b - other.b)<EPSILON && fabs(a - other.a)<EPSILON; }

    __host__ __device__ inline Color4 operator /  (const float f) const { return Color4(r / f, g /f, b / f, a / f);}
    __host__ __device__ inline Color4 operator /= (const float f) { r /= f; g /= f; b /= f; a /= f; return *this;}

    __host__ __device__ inline Color4 operator *  (const float f) const { return Color4(r * f, g *f, b * f, a * f);}
    __host__ __device__ inline Color4 operator *= (const float f) { r *= f; g *= f; b *= f; a *= f; return *this;}

    __host__ __device__ inline Color4 operator +  (const Color4& other) const { return Color4(r + other.r, g + other.g, b + other.b, a + other.a); }
    __host__ __device__ inline Color4 & operator += (const Color4& other) { r += other.r; g += other.g; b += other.b; a += other.a; return *this; }

    __host__ __device__ inline Color4 operator -  (const Color4& other) const { return Color4(r - other.r, g - other.g, b - other.b, a - other.a); }
    __host__ __device__ inline Color4 & operator -= (const Color4& other) { r -= other.r; g -= other.g; b -= other.b; a -= other.a; return *this; }

    __host__ __device__ Luminance toLuminance() {
	return Luminance(0.2126 * r + 0.7152 * g + 0.0722 * b, a);
    }

    float r,g,b,a;
};

struct PatchXF
{
public:
    __host__ __device__ PatchXF ():x(0.f),y(0.f),theta(0.f),hysteresis(0.f),layer(0),scaleU(1.f),scaleV(1.f),luminanceShift(){}
    __host__ __device__ PatchXF (float xx, float yy, float tt, float hh, int ll=0, float su=1.f, float sv=1.f):x(xx),y(yy),theta(tt),hysteresis(hh),layer(ll),scaleU(su),scaleV(sv){}

    __host__ __device__ Color4 toColor4() const { return Color4(x,y,theta,hysteresis); }

    __host__ __device__ float2 xy(float fact=1) const { float2 coord; coord.x = x * fact; coord.y = y * fact; return coord; }

    float x,y,theta,hysteresis;
    int layer;
    float scaleU, scaleV;

    Luminance luminanceShift;
};

struct RibbonP
{
public:
    __host__ __device__ RibbonP ():x(0.f),y(0.f),time_step(0.f),layer(-1){}
    __host__ __device__ RibbonP (float xx, float yy, int tt, int ll=0):x(xx),y(yy),time_step(tt),layer(ll){}

    // For historical reasons the packing order into Color4 is weird.
    __host__ __device__ RibbonP (const Color4 color):x(color.b),y(color.g),time_step(color.r),layer(color.a){}
    __host__ __device__ Color4 toColor4() const { return Color4(time_step,y,x,layer); }

    __host__ __device__ bool isValid() const { return layer >= 0; }

    float x,y;
    int time_step;
    int layer;
};

typedef enum {
    FORWARD = 0,
    BACKWARD,
    BIDIRECTIONAL
} PassDirection;

typedef enum {
    S_FMBM =0,
    S_FMBM_RM,
    S_FBMO,
    S_TCTF,
    S_IND
} SynthesisScheme;

#endif
