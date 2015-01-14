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

#ifndef _DOWNSAMPLE_H_
#define _DOWNSAMPLE_H_

#include <algorithm>
#include <map>
#include <QtCore/QVector>
#include "types.h"

typedef enum {
    DOWNSAMPLE_SCALE_WITH_LEVELS = 0,
    DOWNSAMPLE_SCALE_CONSTANT
} DownsampleScaleMode;

// Specialized template functions to call the different downsampling modes. 
// ImagePyramids with type int using a voting scheme rather than an averaging scheme.
// The float and Color4 definitions are necessary to make the linker happy (yuck).

template <class T> 
void downsamplePyramid(const QVector<T>& input, QList< QVector<T> >& output,
		       int width, int height, DownsampleScaleMode scaleMode);

template <> 
void downsamplePyramid<Color4>(const QVector<Color4>& input, QList< QVector<Color4> >& output,
			       int width, int height, DownsampleScaleMode scaleMode);

template <> 
void downsamplePyramid<float>(const QVector<float>& input, QList< QVector<float> >& output,
			      int width, int height, DownsampleScaleMode scaleMode);

template <>
void downsamplePyramid<float2>(const QVector<float2>& input, QList< QVector<float2> >& output,
			       int width, int height, DownsampleScaleMode scaleMode);

template <>
void downsamplePyramid<float3>(const QVector<float3>& input, QList< QVector<float3> >& output,
                   int width, int height, DownsampleScaleMode scaleMode);

template <> 
void downsamplePyramid<int>(const QVector<int>& input, QList< QVector<int> >& output,
			    int width, int height, DownsampleScaleMode scaleMode);

#endif
