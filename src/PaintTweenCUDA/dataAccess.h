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

#ifndef DATAACCESS_H
#define DATAACCESS_H

#include "types.h"

#include <QtCore/QVector>
#include <QtCore/QStringList>

#include "imagePyramid.h"
#include "texSynth_interface.h"

class NLImageContainer;

//--------------------------------------------------------
// Data
//--------------------------------------------------------

typedef enum {
    IN_INPUT,
    IN_ORIENTATION,
    IN_DIST_TRANS,
    IN_SCALE,
    IN_ID_MERGED,
    IN_SURF_ID,
    IN_VEL_B,
    IN_VEL_F,
    IN_RIBBON_B,
    IN_RIBBON_F,
    NUM_INPUT_ELEMENTS
} InputElements;

typedef enum {
    OUT_OUTPUT,
    OUT_OUTPUT_FINAL,
    OUT_BASE,
    OUT_OFFSETS,
    OUT_OFFSETS_RAW,
    OUT_RESIDUAL,
    OUT_LUMSGHIFT,
    NUM_OUT_ELEMENTS
} OutputElements;

typedef enum {
    STYLE_OUTPUT,
    STYLE_INPUT,
    STYLE_DIST_TRANS,
    STYLE_ORIENTATION,
    STYLE_OBJECT_ID,
    NUM_STYLE_ELEMENTS
} StyleElements;

class DataAccess
{

public:


    //-------------------------------------------------------------------
    // Image input: (frame range starts at 1)
    virtual bool getInputElement(InputElements elt, int frame, ImagePyramid<Color4>& pyramid,
                                 DownsampleScaleMode mode = DOWNSAMPLE_SCALE_CONSTANT) = 0;
    virtual bool getInputElement(InputElements elt, int frame, ImagePyramid<float2>& pyramid,
                                 DownsampleScaleMode mode = DOWNSAMPLE_SCALE_CONSTANT) = 0;
    virtual bool getInputElement(InputElements elt, int frame, ImagePyramid<float>& pyramid,
                                 DownsampleScaleMode mode = DOWNSAMPLE_SCALE_CONSTANT) = 0;
    virtual bool getInputElement(InputElements elt, int frame, ImagePyramid<int>& pyramid) = 0;
    virtual bool getRibbonB(int frame, int step_size, ImagePyramid<float2>& pyramid) = 0;
    virtual bool getRibbonF(int frame, int step_size, ImagePyramid<float2>& pyramid) = 0;
    virtual int getMaxRibbonStep() = 0;
    virtual void getImageDimensions(int& width, int& height) = 0;

    //-------------------------------------------------------------------
    // Output paths:
    virtual QString getOutDir() const = 0;
    virtual QString getTemporaryDir() const = 0;
    virtual QString getOutPath(OutputElements) const = 0;
    virtual QString getStringParameter(const QString& param) const = 0;

    //-------------------------------------------------------------------------------------------------------
    // Styles
    virtual int getNumStyles() const = 0;
    virtual int getStyleWidth(int) const = 0;
    virtual int getStyleHeight(int) const = 0;
    virtual void getMaxStyleDimensions(int& maxWidth, int& maxHeight) const = 0;
    virtual void getStyle(StyleElements element, int i, QVector<Color4>& image) = 0;
    virtual void getStyle(StyleElements element, int i, ImagePyramid<Color4>& pyramid) = 0;
    virtual void getStyle(StyleElements element, int i, ImagePyramid<float>& pyramid) = 0;
    virtual void getStyle(StyleElements element, int i, ImagePyramid<int>& pyramid) = 0;
    virtual void padStyles(int newWidth, int newHeight) = 0;
    virtual const QVector<int>& keyFrameIndices() const = 0;

    //-------------------------------------------------------------------
    // Parameters:	(base class returns defaults):
    virtual float getFloatParameter(const QString&) const = 0;
    virtual int   getIntParameter(const QString&) const = 0;
    virtual bool  getBoolParameter(const QString&) const = 0;

    virtual int firstFrame() const = 0;
    virtual int lastFrame() const = 0;

    virtual const QVector<float>& getOffsetsHistogramSlopes() const = 0;
    virtual const QVector<float>& getOffsetsHistogramThresholds() const = 0;

    virtual TsParameters getTsDefaultParams() const = 0;

    //-------------------------------------------------------------------
    // Animation data and flags:

    virtual PassDirection getFirstPassDirection() const = 0;
    virtual SynthesisScheme getSynthesisScheme() const = 0;
    virtual int getRealtimeSynthesisMode() const { return _realtimeSynthesisMode; }
    virtual int getCurrentPreviewFrame() const { return _curPreviewFrame; }
    virtual int getCurrentPreviewLevel() const { return _curPreviewLevel; }
    virtual int getCurrentPreviewPass() const { return _curPreviewPass; }
    virtual void setCurrentPreviewFrame(int frame){  _curPreviewFrame = frame; }
    virtual void setCurrentPreviewLevel(int level){ _curPreviewLevel = level; }
    virtual void setCurrentPreviewPass(int pass){ _curPreviewPass = pass; }

    static DataAccess& instance() { return *_instance; }

protected:
    // Singleton pattern.
    DataAccess()
        : _useCachedKeyframePreprocess(false),
          _realtimeSynthesisMode(false),
          _curPreviewFrame(1),
          _curPreviewLevel(0),
          _curPreviewPass(0)
    {
        _instance = this;
    }

protected:
    //-------------------------------------------------------------------
    // Animation data and flags:	
    bool _useCachedKeyframePreprocess;
    int  _realtimeSynthesisMode;

    // Current frame for preview:
    int _curPreviewFrame;
    int _curPreviewLevel;
    int _curPreviewPass;

    static DataAccess* _instance;
};

#endif
