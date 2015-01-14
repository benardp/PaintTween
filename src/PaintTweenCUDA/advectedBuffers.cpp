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


#include "advectedBuffers.h"

AdvectedBuffersDevice::AdvectedBuffersDevice(CudaImageBufferDevice<PatchXF> in_o,
                      CudaImageBufferDevice<Color4> in_base,
                      CudaImageBufferDevice<Color4> in_output,
                      CudaImageBufferDevice<float> in_residual,
                      CudaImageBufferDevice<int> in_time_step) :
    offsets(in_o), base(in_base), output(in_output),
    residual(in_residual), timeStep(in_time_step)
{
}

void AdvectedBuffersHost::initialize(int width, int height)
{
    _offsets.initialize(width, height);
    _base_colors.initialize(width, height);
    _output_colors.initialize(width, height);
    _residual.initialize(width, height);
    _time_step.initialize(width, height);
}

void AdvectedBuffersHost::reset()
{
    _offsets.reset();
    _base_colors.reset();
    _output_colors.reset();
    _residual.reset();
    _time_step.reset();
}

void AdvectedBuffersHost::clear()
{
    _offsets.clear();
    _base_colors.clear();
    _output_colors.clear();
    _residual.clear();
    _time_step.clear();
}

void AdvectedBuffersHost::clearHistory()
{
    _offsets.clearHistory();
    _base_colors.clearHistory();
    _output_colors.clearHistory();
    _residual.clearHistory();
    _time_step.clearHistory();
}

void AdvectedBuffersHost::copyCurrentToHistory()
{
    _offsets.copyCurrentToHistory();
    _base_colors.copyCurrentToHistory();
    _output_colors.copyCurrentToHistory();
    _residual.copyCurrentToHistory();
    _time_step.copyCurrentToHistory();
}

void AdvectedBuffersHost::copyHistoryToCurrent(int position)
{
    _offsets.copyHistoryToCurrent(position);
    _base_colors.copyHistoryToCurrent(position);
    _output_colors.copyHistoryToCurrent(position);
    _residual.copyHistoryToCurrent(position);
    _time_step.copyHistoryToCurrent(position);
}

AdvectedBuffersDevice AdvectedBuffersHost::deviceType()
{
    AdvectedBuffersDevice out(_offsets, _base_colors,
                              _output_colors, _residual, _time_step);
    return out;
}
