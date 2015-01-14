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

#ifndef NLDATAACCESS_H
#define NLDATAACCESS_H

#include "PaintTweenCUDA/dataAccess.h"
#include "PaintTweenCUDA/imagePyramid.h"

#include "nlParameters.h"
#include "nlShot.h"
#include "nlStyles.h"

class NLDataAccess : public DataAccess
{
public:
    void clear();

    //-------------------------------------------------------------------
    // Image input:
    bool getInputElement(InputElements elt, int frame, NLImageContainer& image);

    template<class T> bool getInputElement(InputElements elt, int frame, ImagePyramid<T> &pyramid,
                                           const T& defaultValue, DownsampleScaleMode mode = DOWNSAMPLE_SCALE_CONSTANT);
    virtual bool getInputElement(InputElements elt, int frame, ImagePyramid<Color4>& pyramid,
                                 DownsampleScaleMode mode = DOWNSAMPLE_SCALE_CONSTANT);
    virtual bool getInputElement(InputElements elt, int frame, ImagePyramid<float2>& pyramid,
                                 DownsampleScaleMode mode = DOWNSAMPLE_SCALE_CONSTANT);
    virtual bool getInputElement(InputElements elt, int frame, ImagePyramid<float>& pyramid,
                                 DownsampleScaleMode mode = DOWNSAMPLE_SCALE_CONSTANT);
    virtual bool getInputElement(InputElements elt, int frame, ImagePyramid<int>& pyramid);
    virtual bool getRibbonB(int frame, int step_size, ImagePyramid<float2>& pyramid);
    virtual bool getRibbonF(int frame, int step_size, ImagePyramid<float2>& pyramid);
    virtual int getMaxRibbonStep();
    virtual void getImageDimensions(int& width, int& height);

    //-------------------------------------------------------------------------------------------------------
    // Parameters
    virtual int firstFrame() const;
    virtual int lastFrame() const;
    void setFirstFrame(int frame);
    void setLastFrame(int frame);

    virtual SynthesisScheme getSynthesisScheme() const;
    virtual PassDirection getFirstPassDirection() const;

    virtual float getFloatParameter(const QString& param) const;
    virtual int   getIntParameter(const QString& param) const;
    virtual bool  getBoolParameter(const QString& param) const;

    virtual const QVector<float>& getOffsetsHistogramSlopes() const;
    virtual const QVector<float>& getOffsetsHistogramThresholds() const;

    virtual TsParameters getTsDefaultParams() const;

    //-------------------------------------------------------------------------------------------------------
    // Styles
    virtual int getNumStyles() const;
    virtual int getStyleWidth(int i) const;
    virtual int getStyleHeight(int) const;
    virtual void getMaxStyleDimensions(int &maxWidth, int &maxHeight) const;

    virtual void getStyle(StyleElements element, int i, QVector<Color4>& style_img);
    virtual void getStyle(StyleElements element, int i, ImagePyramid<Color4>& pyramid);
    virtual void getStyle(StyleElements element, int i, ImagePyramid<float>& pyramid);
    virtual void getStyle(StyleElements element, int i, ImagePyramid<int>& pyramid);
    virtual void padStyles(int newWidth, int newHeight);
    virtual const QVector<int>& keyFrameIndices() const;

    void getStyle(StyleElements element, int i, NLImageContainer &image);

    //-------------------------------------------------------------------
    // Output paths:
    virtual QString getOutDir() const;
    virtual QString getTemporaryDir() const;
    virtual QString getOutPath(OutputElements element) const;
    virtual QString getStringParameter(const QString& param) const { return _stringParameters.value(param, ""); }
    void setStringParameter(const QString& param, const QString& value) { _stringParameters[param] = value; }

    QString getWorkingDir() const;
    int version() const;

    inline void setOutputDir(const QString& outputDir) {_outputDir = outputDir; }
    inline void setTemporaryDir(const QString& temporaryDir) {_temporaryDir = temporaryDir; }
    inline void setStoreIntermediateImagesInTemp(bool enable) { _storeIntermediateImagesInTemp = enable; }

    bool goToFrame(int frame);

protected:
    // Singleton pattern. The instance itself is an NLSynthesizer.
    NLDataAccess();

    // Pyramid conversions
    void toOrientationPyramid(const ImagePyramid<float3>& fullPyramid, ImagePyramid<float> &pyramid);
    QString pyramidPath(InputElements elt, int frame) const;
    QString pyramidPath(StyleElements elt) const;

    bool isStyleReady(QString& msg) const;
    bool isAnimReady(QString &msg) const;

    NLShot _shot;
    NLStyles _styles;

    bool _dataUptodate;
    bool _loaded;

protected:
    // Output directory:
    QString _outputDir;
    QString _temporaryDir;
    QHash<QString,QString> _stringParameters;
    bool _storeIntermediateImagesInTemp;

    NLImageContainer _blank;

    int _firstFrame, _lastFrame;
};

#endif // NLDATAACCESS_H
