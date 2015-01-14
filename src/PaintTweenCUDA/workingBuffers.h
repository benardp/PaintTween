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

#ifndef WORKING_BUFFERS_H
#define WORKING_BUFFERS_H

#include "cudaImageBuffer.h"
#include "advectedBuffers.h"
#include "types.h"

#include "texSynth_interface.h"

#include <string>
#include <QtCore/QList>
#include <QtCore/QString>

class WorkingBuffers
{   
public:
    WorkingBuffers() :
        _is_initialized(false),
        _image_width(0), _image_height(0),
        _style_width(0), _style_height(0),
        _history_position(0) {}

    void initialize();
    bool isInitialized() { return _is_initialized; }
    void clear();
    void resetAdvectedBuffers();
    void setCurrentLevel(int level) { _current_level = level; }

    Color4 outputPixel(int x, int y);
    float4 residualPixel(int x, int y);
    PatchXF offsetsPixel(int x, int y);
    float  histogramPixel(int x, int y);
    int currentLevel() const;
    int currentImageWidth() const;
    int currentImageHeight() const;
    int currentStyleWidth(int styleIndex) const;
    int currentStyleHeight(int styleIndex) const;

    void getOutputDimensions(int level, int* width, int* height);
    void getStyleDimensions(int level, int* width, int* height);

    bool copyToGL(struct cudaGraphicsResource* destination, TsOutputType outputType, int level, int styleIndex,
                  float vizGain = 1.0f, bool vizNormalize = true, int vizMode = 0);

    bool loadOutput(int level, int frame, int pass);
    bool loadOffsets(int level, int frame, int pass);
    bool loadResidual(int level, int frame, int pass);

    bool saveTempImages(int level, int frame, int pass, int version);
    bool linkTempImagesToNextPass(int level, int frame, int source_pass, int this_pass);
    bool loadTempImages(int level, int frame, int pass);
    bool saveFinalOutput(const QString& outputPath);
    void saveOffsets(int level, const QString& filePath);
    void saveDebugImages(int level, int frame, int pass, int version);

    void setParams(const TsParameters& params) { _params = params; }
    TsParameters& params() { return _params;}

    void clearHistory();
    void takeHistorySnapshot(const QString& message);
    int historyPosition() const { return _history_position; }
    int historySize() const { return _history_params.size(); }
    bool stepHistoryBack() { return gotoHistoryPosition(_history_position+1); }
    bool stepHistoryForward() { return gotoHistoryPosition(_history_position-1); }

public:
    // Image-size buffers
    CudaBufferPair<PatchXF>      offsets;
    CudaBufferPair<float>        residualCache;

    ///// Sometimes useful for debugging.
    //CudaImageBufferHost<Color4>  canvasInputCache;
    /////

    CudaImageBufferHost<Color4>  canvasOutputCache;
    CudaImageBufferHost<float2>  canvasVelFCache;
    CudaImageBufferHost<float2>  canvasVelBCache;
    CudaImageBufferHost<float2>  distTransCache;

    CudaBufferPair<RibbonP>     ribbonField;

    AdvectedBuffersHost advectedF;
    AdvectedBuffersHost advectedB;

    // Style-size buffers
    CudaBufferPair<float>       offsetsHistogram;

    // Temporary buffers
    CudaImageBufferHost<Color4> tempOutput;
    CudaImageBufferHost<Color4> tempOutputHistogram;

protected:
    void getOutput(int level, QVector<PatchXF>& out_offsets, QVector<Color4>& output,
		   QVector<float>& residual, QVector<float>& histogram);
    void copyOutputArray(TsOutputType outputType, int level, cudaArray* output, int styleIndex,
                         float vizGain, bool vizNormalize, int vizMode);

    bool gotoHistoryPosition(int position);

    void colorMapOffsets(const QVector<PatchXF>& input, QVector<Color4>& output);

protected:
    bool _is_initialized;
    int _image_width, _image_height;
    int _style_width, _style_height;
    int _current_level;

    typedef struct {
        int level;
        QString message;
    } HistoryParams;
    QList<HistoryParams> _history_params;
    int  _history_position;

    // All the constant parameters as defined in texSynth_interface.h
    TsParameters _params;
};

#endif // WORKING_BUFFERS_H
