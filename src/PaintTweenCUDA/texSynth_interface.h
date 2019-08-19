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

#ifndef _TS_INTERFACE_H_
#define _TS_INTERFACE_H_

#include "types.h"

#define NUM_MAX_STYLES 5

typedef enum {
    TS_OUTPUT_CANVAS = 0,
    TS_OUTPUT_CANVAS_INPUT,
    TS_OUTPUT_OFFSET,
    TS_OUTPUT_RESIDUAL,
    TS_OUTPUT_LUMINANCESHIFT,
    TS_OUTPUT_VEL_F,
    TS_OUTPUT_VEL_B,
    TS_OUTPUT_ADVECTED_F,
    TS_OUTPUT_ADVECTED_B,
    TS_OUTPUT_HISTOGRAM,
    TS_OUTPUT_DIST_TRANS,
    TS_OUTPUT_ID,
    TS_OUTPUT_RIBBON_F,
    TS_OUTPUT_RIBBON_B,
    TS_OUTPUT_INVALID
} TsOutputType;

typedef enum {
    TS_LAYER_EXEMPLAR_BASE = 0,
    TS_LAYER_EXEMPLAR_OUTPUT,
    TS_LAYER_EXEMPLAR_ORIENTATION,
    TS_LAYER_EXEMPLAR_DIST_TRANS,
    TS_LAYER_EXEMPLAR_OBJECT_IDS,
    TS_LAYER_INPUT_COLOR,
    TS_LAYER_INPUT_ORIENTATION,
    TS_LAYER_INPUT_DIST_TRANS,
    TS_LAYER_INPUT_SCALE,
    TS_LAYER_INPUT_ID,
    TS_LAYER_INPUT_SURF_ID,
    TS_LAYER_INPUT_VEL_B,
    TS_LAYER_INPUT_VEL_F,
    TS_LAYER_INPUT_VEL_B_PREVIOUS,
    TS_LAYER_INPUT_VEL_F_PREVIOUS,
    TS_LAYER_GUIDE_B,
    TS_LAYER_GUIDE_F,
    TS_LAYER_CUMUL_FIELD_B,
    TS_LAYER_CUMUL_FIELD_F,
    TS_LAYER_FRAME_TO_KEY_RIBBON_B,
    TS_LAYER_FRAME_TO_KEY_RIBBON_F,
    TS_LAYER_RANDOM,
    TS_LAYER_INVALID
} TsImageLayer;

typedef struct
{
    int residualWindowSize;
    float coherenceWeight;
    float distTransWeight;

    // Luminance shift
    float lumShiftWeight;
    bool lumShiftEnable;

    float inputAnalogyWeight;
    float canvasOutputWeight;

    int numStyles;

    // offsets histogram slope and intensity threshold
    float offsetsHistogramSlope[NUM_MAX_STYLES];
    float offsetsHistogramThreshold[NUM_MAX_STYLES];

    // max size (in x or y direction) a pixel can map to following a velocity
    // transform before it is considered invalid
    // NOTE: this is at the finest resolution, it is rescaled down for other levels
    float maxDistortion;

    // how much more to weight the alpha channel 
    float alphaWeight;
    bool transparencyOk;

    // how much we should penalize pixels in the image
    // coming from a pixel in the style with a different object id:
    float styleObjIdWeight;

    // if false, suppress orientation when computing residual
    bool orientationEnabled;

    // colorspace (RGB or LAB)
    int colorspace;

    // coherence parameters: ratio and angle
    float coherenceRatioRange;
    float coherenceAngleRange;

    // levels in the pyramid and max scale when using multi-res neighbourhood
    int levels;

    int scatterSamplesCount;

    float advectionJitter;

    // Time derivative weights
    float timeDerivativeInputWeight;
    float timeDerivativeOutputWeight;
    float temporalCoherenceWeight;
    float hysteresisWeight;

    // Guided key-frame interpolation
    bool interpolateKeyFrame;
    bool useGuide;
    int  currentInterpolationIndex;
    int  numFrames;
    int  numKeyFrames;
    int  firstKeyIndex;
    int  lastKeyIndex;

    // A culling test to avoid wasting search time on uninteresting pixels.
    // anything outside this distance is ignored.
    int distTransCull;

    // Global scaling parameters for the residual neighborhood.
    float residualWindowScaleU, residualWindowScaleV;

    PassDirection direction;
 
} TsParameters;

#endif
