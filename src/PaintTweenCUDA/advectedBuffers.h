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


#ifndef ADVECTED_BUFFERS_H
#define ADVECTED_BUFFERS_H

#include "cudaImageBuffer.h"
#include "types.h"

class AdvectedBuffersDevice
{
public:
    __device__ AdvectedBuffersDevice(CudaImageBufferDevice<PatchXF> in_o,
                          CudaImageBufferDevice<Color4> in_base,
                          CudaImageBufferDevice<Color4> in_output,
                          CudaImageBufferDevice<float> in_residual,
                          CudaImageBufferDevice<int> in_time_step);

public:
    CudaImageBufferDevice<PatchXF> offsets;
    CudaImageBufferDevice<Color4> base;
    CudaImageBufferDevice<Color4> output;
    CudaImageBufferDevice<float> residual;
    CudaImageBufferDevice<int> timeStep;
};

class AdvectedBuffersHost
{
public:
    void initialize(int width, int height);
    void reset();
    void clear();

    void clearHistory();
    void copyCurrentToHistory();
    void copyHistoryToCurrent(int position);

    AdvectedBuffersDevice deviceType();
    operator AdvectedBuffersDevice() { return deviceType(); }

    CudaImageBufferHost<PatchXF>& offsets() { return _offsets; }
    CudaImageBufferHost<Color4>& base() { return _base_colors; }
    CudaImageBufferHost<Color4>& output() { return _output_colors; }
    CudaImageBufferHost<float>& residual() { return _residual; }
    CudaImageBufferHost<int>& timeStep() { return _time_step; }

protected:
    CudaImageBufferHost<PatchXF> _offsets;
    CudaImageBufferHost<Color4> _base_colors;
    CudaImageBufferHost<Color4> _output_colors;
    CudaImageBufferHost<float> _residual;
    CudaImageBufferHost<int> _time_step;

};

#endif // ADVECTED_BUFFERS_H
