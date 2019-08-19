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

#ifndef __TEXSYNTH_KERNEL_H_
#define __TEXSYNTH_KERNEL_H_

#include "texSynth_interface.h"
#include "imagePyramid.h"

#include "workingBuffers.h"

#include <vector>

namespace TexSynth {

// These calls should really be a single template function, but the template function
// makes the linker unhappy.
//void uploadLUT(const vector<float>& lut);
void initialize_images(int width, int height);
void initialize_exemplars(int width, int height, int numLayers);
void initialize_cumulativeFields(int width, int height, int numLayers);
void uploadImage_float2(int level, TsImageLayer layer, const ImagePyramid<float2>& image);
void uploadImage_Color4(int level, TsImageLayer layer, const ImagePyramid<Color4>& image, int styleLayer=0);
void uploadImage_int(int level, TsImageLayer layer, const ImagePyramid<int>& image, int styleLayer=0);
void uploadImage_float(int level, TsImageLayer layer, const ImagePyramid<float>& image, int styleLayer=0);
void uploadParameters(const TsParameters &params);
void flipAnimBuffers();

void randomizeOffsets(WorkingBuffers* buffers, int level);
void offsetsFromFrameToKeyRibbon(WorkingBuffers* buffers, int level, bool look_forwards);

void propagate(WorkingBuffers* buffers, int level, int dist, int keyIndex);
void scatter(WorkingBuffers* buffers, int level, int keyIndex);

void upsample(WorkingBuffers* buffers, int fine_level);

void advectOffsets(WorkingBuffers* buffers, int level, int time_step, bool forwards, int op_flags);

void accumulateRibbonField(WorkingBuffers* buffers, int level, bool forwards,
                           int time_step, bool overwrite);
void mergeUsingResidual(WorkingBuffers* buffers, int level, bool forwards);
void mergeRandomly(WorkingBuffers* buffers, int level, bool forwards, float weightAdvected, bool force_valid);

void mergeAdvectedBuffers(WorkingBuffers* buffers, int level, int merge_fallback);

void updateResidualCache(WorkingBuffers* buffers, int level);
void syncBuffers(WorkingBuffers* buffers, int level, int op_flags = 0);

void setup(TsParameters params);

void copyInput(TsImageLayer layer, cudaArray* array, int width, int height, int styleLayer=0);

const void *getTextureReferenceByName( const char *name);

enum {
    DONT_COPY_OFFSETS = 0x01,
    DONT_UPDATE_RESIDUAL = 0x10,
    MERGE_FALLBACK_RANDOM = 0x20,
    MERGE_FALLBACK_LAST_PASS = 0x40
};

enum {
    ADVECT_IGNORE_RANDOM = 0x1,
    ADVECT_DO_NOT_CHECK_PREVIOUS = 0x10,
    ADVECT_CHECK_FRAME_TO_KEY = 0x20,
    ADVECT_UPDATE_RESIDUAL = 0x80
};

}

#endif
