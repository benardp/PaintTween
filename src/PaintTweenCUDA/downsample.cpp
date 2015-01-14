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

#include <cmath>
#include <assert.h>
#include <QtCore/QHash>
#include <QtCore/QMap>
#include <QtCore/QVector>

#include "downsample.h"
#include "cutil_math.h"
#include "cudaHostUtil.h"

// Function to downsample the input and store it
// in a hierarchy of resolutions (all in one image):

template <class T>
void downsampleAverage(const QVector<T>& input,
		       QList< QVector<T> >& output,
		       int width, int height,
		       DownsampleScaleMode scaleMode = DOWNSAMPLE_SCALE_CONSTANT)
{
    int levels = std::min(floor(log2f(width)), floor(log2f(height)));

    output.append(QVector<T>(input));
    int prevlWidth = width;

    for(int i=1; i<levels; i++){
	int lWidth = width >> i;
	int lHeight = height >> i;

	QVector<T> lOutput(lWidth * lHeight);
	T* data = lOutput.data();
	T* prev = output.last().data();

	for(int y=0; y < lHeight; y++){
	    for(int x=0; x < lWidth; x++){

		T sum = T(0.f);
		sum += prev[ y*2*prevlWidth    + x*2];
		sum += prev[ y*2*prevlWidth    + x*2+1];
		sum += prev[(y*2+1)*prevlWidth + x*2];
		sum += prev[(y*2+1)*prevlWidth + x*2+1];

		sum *= 0.25f;
		if(scaleMode == DOWNSAMPLE_SCALE_WITH_LEVELS){
		    sum *= 0.5f;
		}

		data[y*lWidth+x] = sum;
	    }
	}
	output.append(lOutput);
	prevlWidth = lWidth;
    }
}

void downsampleAverage(const QVector<float2>& input,
		       QList< QVector<float2> >& output,
		       int width, int height,
		       DownsampleScaleMode scaleMode = DOWNSAMPLE_SCALE_CONSTANT)
{
    int levels = std::min(floor(log2f(width)), floor(log2f(height)));

    output.append(QVector<float2>(input));
    int prevlWidth = width;

    for(int i=1; i<levels; i++){
	int lWidth = width >> i;
	int lHeight = height >> i;

	QVector<float2> lOutput(lWidth * lHeight);
	float2* data = lOutput.data();
	float2* prev = output.last().data();

	for(int y=0; y < lHeight; y++){
	    for(int x=0; x < lWidth; x++){

		float2 sum = make_float2(0.f,0.f);
		sum += prev[ y*2*prevlWidth    + x*2];
		sum += prev[ y*2*prevlWidth    + x*2+1];
		sum += prev[(y*2+1)*prevlWidth + x*2];
		sum += prev[(y*2+1)*prevlWidth + x*2+1];

		sum *= 0.25f;
		if(scaleMode == DOWNSAMPLE_SCALE_WITH_LEVELS){
		    sum *= 0.5f;
		}

		data[y*lWidth+x] = sum;
	    }
	}
	output.append(lOutput);
	prevlWidth = lWidth;
    }
}

void downsampleAverage(const QVector<float3>& input,
               QList< QVector<float3> >& output,
               int width, int height,
               DownsampleScaleMode scaleMode = DOWNSAMPLE_SCALE_CONSTANT)
{
    int levels = std::min(floor(log2f(width)), floor(log2f(height)));

    output.append(QVector<float3>(input));
    int prevlWidth = width;

    for(int i=1; i<levels; i++){
    int lWidth = width >> i;
    int lHeight = height >> i;

    QVector<float3> lOutput(lWidth * lHeight);
    float3* data = lOutput.data();
    float3* prev = output.last().data();

    for(int y=0; y < lHeight; y++){
        for(int x=0; x < lWidth; x++){

        float3 sum = make_float3(0.f,0.f,0.f);
        sum += prev[ y*2*prevlWidth    + x*2];
        sum += prev[ y*2*prevlWidth    + x*2+1];
        sum += prev[(y*2+1)*prevlWidth + x*2];
        sum += prev[(y*2+1)*prevlWidth + x*2+1];

        sum *= 0.25f;
        if(scaleMode == DOWNSAMPLE_SCALE_WITH_LEVELS){
            sum *= 0.5f;
        }

        data[y*lWidth+x] = sum;
        }
    }
    output.append(lOutput);
    prevlWidth = lWidth;
    }
}

template <class T>
void downsampleVote(const QVector<T>& input, QList< QVector<T> >& output, int width, int height)
{
    int levels = std::min(floor(log2f(width)), floor(log2f(height)));

    output.append(QVector<T>(input));
    int prevlWidth = width;

    for(int i=1; i<levels; i++){
	int lWidth = width >> i;
	int lHeight = height >> i;

	QVector<T> lOutput(lWidth * lHeight);
	T* data = lOutput.data();
	T* prev = output.last().data();

	for(int y=0; y < lHeight; y++){
	    for(int x=0; x < lWidth; x++){

                T values[4];
                QHash<T,int> valuesAndScores;

                // store the (possibly unique) value at each pixel
		values[0] = prev[y*2*prevlWidth + x*2];
		values[1] = prev[y*2*prevlWidth + x*2+1];
		values[2] = prev[(y*2+1)*prevlWidth + x*2];
		values[3] = prev[(y*2+1)*prevlWidth + x*2+1];

                // count the number of appearances of each pixel
                for(unsigned int j = 0; j < 4; j++) {
                    valuesAndScores.insert(values[j],valuesAndScores.value(values[j],0)+1);
                }

                QMap<int,T> scoresAndValues;
                QHashIterator<T,int> itHash(valuesAndScores);
                while(itHash.hasNext()){
                    itHash.next();
                    scoresAndValues.insert(itHash.value(),itHash.key());
                }

                QMapIterator<int,T> itMap(scoresAndValues); itMap.toBack(); // best score
		data[y*lWidth+x] = itMap.previous().value();
            }
        }
	output.append(lOutput);
	prevlWidth = lWidth;
    }
}

template <class T> 
void downsamplePyramid(const QVector<T>& input, QList< QVector<T> >& output,
		       int width, int height, DownsampleScaleMode scaleMode) {
    downsampleAverage(input, output, width, height, scaleMode);
}

template <> 
void downsamplePyramid<Color4>(const QVector<Color4>& input, QList< QVector<Color4> >& output,
			       int width, int height, DownsampleScaleMode scaleMode) {
    downsampleAverage(input, output, width, height, scaleMode);
}

template <> 
void downsamplePyramid<float>(const QVector<float>& input,QList<  QVector<float> >& output,
			      int width, int height, DownsampleScaleMode scaleMode) {
    downsampleAverage(input, output, width, height, scaleMode);
}

template <>
void downsamplePyramid<int>(const QVector<int>& input, QList< QVector<int> >& output,
			    int width, int height, DownsampleScaleMode scaleMode) {
    Q_UNUSED(scaleMode);
    downsampleVote(input, output, width, height);
}

template <>
void downsamplePyramid<float2>(const QVector<float2>& input, QList< QVector<float2> >& output,
			       int width, int height, DownsampleScaleMode scaleMode) {
    downsampleAverage(input, output, width, height, scaleMode);
}

template <>
void downsamplePyramid<float3>(const QVector<float3>& input, QList< QVector<float3> >& output,
                   int width, int height, DownsampleScaleMode scaleMode) {
    downsampleAverage(input, output, width, height, scaleMode);
}
