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

#include "workingBuffers.h"

#if _MSC_VER
#include <math.h>
#define isnan(x) _isnan(x)
#define isinf(x) (!_finite(x))
#define fpu_error(x) (isinf(x) || isnan(x))
#endif

#include "imageIO.h"

#include "texSynth_interface.h"
#include "texSynth_kernel.h"
#include "dataAccess.h"

#include <iostream>
#include <cstdio>
#include <math.h>
#include <QtNetwork/QHostInfo>

void WorkingBuffers::initialize()
{
    clearHistory();
    DataAccess& data_access = DataAccess::instance();

    int imageWidth, imageHeight;
    int styleWidth, styleHeight;
    data_access.getImageDimensions(imageWidth, imageHeight);
    data_access.getMaxStyleDimensions(styleWidth, styleHeight);
    styleHeight *= data_access.getNumStyles();

    bool is_right_size =
            (int)imageWidth == _image_width && (int)imageHeight == _image_height &&
            (int)styleWidth == _style_width && (int)styleHeight == _style_height;

    if (_is_initialized && is_right_size) {
        return;
    }
    std::cout << "Initializing CUDA working memory ("
              << imageWidth << " " << imageHeight << " "
              << styleWidth << " " << styleHeight << ")" << std::endl;

    // Image-size buffers
    offsets.initialize(imageWidth, imageHeight);
    residualCache.initialize(imageWidth, imageHeight);
    canvasOutputCache.initialize(imageWidth, imageHeight);
    //canvasInputCache.initialize(imageWidth, imageHeight);
    canvasVelFCache.initialize(imageWidth, imageHeight);
    canvasVelBCache.initialize(imageWidth, imageHeight);
    distTransCache.initialize(imageWidth, imageHeight);
    ribbonField.initialize(imageWidth, imageHeight);

    advectedF.initialize(imageWidth, imageHeight);
    advectedB.initialize(imageWidth, imageHeight);

    tempOutput.initialize(imageWidth, imageHeight);

    // Style-size buffers
    offsetsHistogram.initialize(styleWidth, styleHeight);

    _is_initialized = true;
    _style_width = styleWidth;
    _style_height = styleHeight;
    _image_width = imageWidth;
    _image_height = imageHeight;
    _current_level = 0;
}

void WorkingBuffers::clear()
{
    if (!_is_initialized)
        return;

    offsets.clear();
    residualCache.clear();
    //canvasInputCache.clear();
    canvasOutputCache.clear();
    canvasVelFCache.clear();
    canvasVelBCache.clear();
    distTransCache.clear();
    ribbonField.clear();
    offsetsHistogram.clear();
    tempOutput.clear();
    tempOutputHistogram.clear();

    advectedF.clear();
    advectedB.clear();

    _is_initialized = false;
}

void WorkingBuffers::resetAdvectedBuffers()
{
    advectedF.reset();
    advectedB.reset();
}

void WorkingBuffers::getOutput(int level, QVector<PatchXF>& out_offsets, QVector<Color4>& output,
			  QVector<float>& residual, QVector<float>& histogram)
{
    if (!_is_initialized) {
        std::cout << "ERROR: attempting to read when cuda memory is not allocated." << std::endl;
        return;
    }

    offsets.base().copyToHost(out_offsets, level);
    residualCache.base().copyToHost(residual, level);

    ////
    canvasOutputCache.copyToHost(output, level);
    //canvasInputCache.copyToHost(output, level);
    ////

    offsetsHistogram.base().copyToHost(histogram, level);
}

void WorkingBuffers::getOutputDimensions(int level, int* width, int* height)
{
    if (!_is_initialized) {
        std::cout << "ERROR: attempting to read when cuda memory is not allocated." << std::endl;
        return;
    }

    *width = offsets.width(level);
    *height = offsets.height(level);
}

void WorkingBuffers::getStyleDimensions(int level, int* width, int* height)
{
    if (!_is_initialized) {
        std::cout << "ERROR: attempting to read when cuda memory is not allocated." << std::endl;
        return;
    }

    *width = offsetsHistogram.width(level);
    *height = offsetsHistogram.height(level);
}

bool WorkingBuffers::loadResidual(int level, int frame, int pass)
{
    QVector<Color4> residualI;
    QVector<float> residual;
    int width;
    int height;
    QString path = ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_RESIDUAL),level,frame,pass,0);
    if(!ImageIO::readImage(path, residualI, width, height)){
        std::cerr << "Residual load error! " << std::endl;
        return false;
    }
    if ((int)width != residualCache.width(level) || (int)height != residualCache.height(level)) {
        return false;
    }

    residual.resize(residualI.size());
    for(int i=0; i<residualI.size(); i++){
        residual[i] = residualI[i].b;
    }
    residualCache.base().copyFromHost(residual, level);

    return true;
}

bool WorkingBuffers::loadOffsets(int level, int frame, int pass)
{
    QVector<PatchXF> offsets_vec;
    int offsetsWidth;
    int offsetsHeight;
    QString path = ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_OFFSETS_RAW),level,frame,pass,0);
    if(!ImageIO::readImage(path, offsets_vec, offsetsWidth, offsetsHeight)){
        std::cerr << "Offsets load error! " << std::endl;
        return false;
    }
    if ((int)offsetsWidth != offsets.width(level) || (int)offsetsHeight != offsets.height(level)) {
        return false;
    }

    offsets.base().copyFromHost(offsets_vec, level);

    return true;
}

bool WorkingBuffers::loadOutput(int level, int frame, int pass)
{
    QVector<Color4> output;
    int outputWidth;
    int outputHeight;
    QString path = ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_OUTPUT),level,frame,pass,0);
    if(!ImageIO::readImage(path, output, outputWidth, outputHeight)){
        return false;
    }
    if ((int)outputWidth != canvasOutputCache.width(level) || (int)outputHeight != canvasOutputCache.height(level)) {
        return false;
    }

    canvasOutputCache.copyFromHost(output, level);

    return true;
}

void WorkingBuffers::colorMapOffsets(const QVector<PatchXF>& input, QVector<Color4>& output)
{
    int exemplarBaseWidth = _style_width;
    int exemplarBaseHeight = _style_height;
    int numStyles = params().numStyles;

    output.resize(input.size());
    for (int i = 0; i < output.size(); i++) {
        output[i].r = input[i].x / (float)exemplarBaseWidth;
        output[i].g = (input[i].y + input[i].layer * exemplarBaseHeight) / (float)(exemplarBaseHeight * numStyles);
        output[i].b = input[i].theta;
        output[i].a = input[i].hysteresis;
    }
}

bool WorkingBuffers::saveTempImages(int level, int frame, int pass, int version)
{
    QVector<PatchXF> offsets;
    QVector<Color4> output;
    QVector<float> residual;
    QVector<float> histogram;

    int outputWidth, outputHeight;

    _current_level = level;

    getOutput(level, offsets,output,residual,histogram);
    getOutputDimensions(level, &outputWidth, &outputHeight);

    QVector<Color4> residualOut(offsets.size());
    QVector<Color4> luminanceShiftOut(offsets.size());

    assert(residual.size() == residualOut.size());
    assert(residual.size() == offsets.size());

    float totalResidual = 0.f;

    for(int i=0; i<residual.size(); i++){
        residualOut[i].r = residual[i]*0.0001f;
        residualOut[i].g = residual[i]*0.01f;
        residualOut[i].b = residual[i];
        residualOut[i].a = 1.f;

        totalResidual += residual[i];

        luminanceShiftOut[i].r = offsets[i].luminanceShift.y > 0 ? offsets[i].luminanceShift.y : 0;
        luminanceShiftOut[i].g = offsets[i].luminanceShift.y < 0 ? -offsets[i].luminanceShift.y : 0;
        luminanceShiftOut[i].b = offsets[i].luminanceShift.a > 0 ? 1 : 0;
        luminanceShiftOut[i].a = 1.f;
    }

    QVector<Color4> offsetsOut;
    colorMapOffsets(offsets, offsetsOut);

    if (isnan(totalResidual)) {
        qCritical("WARNING: Residual contains NaNs!");
    }

    /*QString residual_file = DataAccess::instance().getTemporaryDir() + "/residual.csv";
    FILE* fp = fopen(qPrintable(residual_file), "a");
    fprintf(fp,"%d %f\n",frame,totalResidual);
    fclose(fp);*/

    ImageIO::writeImage(output, ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_OUTPUT),level, frame, pass, version), outputWidth, outputHeight);
    ImageIO::writeImage(offsetsOut, ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_OFFSETS),level, frame, pass, version), outputWidth, outputHeight);
    ImageIO::writeImage(offsets, ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_OFFSETS_RAW),level, frame, pass, version), outputWidth, outputHeight);
    ImageIO::writeImage(residualOut, ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_RESIDUAL),level, frame, pass, version), outputWidth, outputHeight);

    return true;
}

void WorkingBuffers::saveDebugImages(int level, int frame, int pass, int version)
{
    DataAccess& data = DataAccess::instance();
    QVector<PatchXF> out_offsets, advected_f_offsets, advected_b_offsets;

    if (!_is_initialized) {
        std::cout << "ERROR: attempting to read when cuda memory is not allocated." << std::endl;
        return;
    }

    offsets.base().copyToHost(out_offsets, level);
    advectedF.offsets().copyToHost(advected_f_offsets, level);
    advectedB.offsets().copyToHost(advected_b_offsets, level);

    QVector<Color4> out_this, out_f, out_b;

    colorMapOffsets(out_offsets, out_this);
    colorMapOffsets(advected_f_offsets, out_f);
    colorMapOffsets(advected_b_offsets, out_b);

    int outputWidth, outputHeight;
    getOutputDimensions(level, &outputWidth, &outputHeight);

    QString path = data.getTemporaryDir() + "/offsetsdebug%1.%2.exr";
    ImageIO::writeImage(out_this, ImageIO::temporaryOutputPath(path, level, frame, pass, version), outputWidth, outputHeight);
    path = data.getTemporaryDir() + "/advfdebug%1.%2.exr";
    ImageIO::writeImage(out_f, ImageIO::temporaryOutputPath(path, level, frame, pass, version), outputWidth, outputHeight);
    path = data.getTemporaryDir() + "/advbdebug%1.%2.exr";
    ImageIO::writeImage(out_b, ImageIO::temporaryOutputPath(path, level, frame, pass, version), outputWidth, outputHeight);
}

bool WorkingBuffers::linkTempImagesToNextPass(int level, int frame, int source_pass, int this_pass)
{
    QString source_out = ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_OUTPUT),level, frame, source_pass, 0);
    QString source_offsets = ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_OFFSETS),level, frame, source_pass, 0);
    QString source_offsets_raw = ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_OFFSETS_RAW),level, frame, source_pass, 0);
    QString source_residual = ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_RESIDUAL),level, frame, source_pass, 0);

    QString this_out = ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_OUTPUT),level, frame, this_pass, 0);
    QString this_offsets = ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_OFFSETS),level, frame, this_pass, 0);
    QString this_offsets_raw = ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_OFFSETS_RAW),level, frame, this_pass, 0);
    QString this_residual = ImageIO::temporaryOutputPath(DataAccess::instance().getOutPath(OUT_RESIDUAL),level, frame, this_pass, 0);

    // Non-home (like /scratch) links have to be munged to be addressable over the network.
    QString hostname = QHostInfo::localHostName();
    source_out = ImageIO::netAddressablePath(hostname, source_out);
    source_offsets = ImageIO::netAddressablePath(hostname, source_offsets);
    source_offsets_raw = ImageIO::netAddressablePath(hostname, source_offsets_raw);
    source_residual = ImageIO::netAddressablePath(hostname, source_residual);

    // Have to remove first because ::link won't overwrite an existing file.
    QFile::remove(this_out);
    QFile::remove(this_offsets);
    QFile::remove(this_offsets_raw);
    QFile::remove(this_residual);

    bool success = true;
    success = success && QFile::link(source_out, this_out);
    success = success && QFile::link(source_offsets, this_offsets);
    success = success && QFile::link(source_offsets_raw, this_offsets_raw);
    success = success && QFile::link(source_residual, this_residual);
    if (!success) {
        qDebug("Failure to create links at level %d frame %d\n", level, frame);
    }

    return success;
}

bool WorkingBuffers::loadTempImages(int level, int frame, int pass)
{
    if (!_is_initialized)
        initialize();

    // Zero out everything
    clearHistory();
    offsets.reset();
    residualCache.reset();
    canvasOutputCache.reset();
    //canvasInputCache.reset();
    canvasVelFCache.reset();
    canvasVelBCache.reset();
    distTransCache.reset();
    offsetsHistogram.reset();

    advectedF.reset();
    advectedB.reset();

    // Load the buffers that we save to disk.
    if (!loadOutput(level, frame, pass))
        return false;
    loadOffsets(level, frame, pass);
    loadResidual(level, frame, pass);

    _current_level = level;

    return true;
}

bool WorkingBuffers::saveFinalOutput(const QString& outputPath)
{
    if (!_is_initialized)
        return false;

    QVector<PatchXF> offsets;
    QVector<Color4> output;
    QVector<float> residual;
    QVector<float> histogram;

    getOutput(0, offsets,output,residual,histogram);

    ImageIO::writeImage(output, outputPath, _image_width, _image_height);

    return true;
}

void WorkingBuffers::saveOffsets(int level, const QString& filePath)
{
    QVector<PatchXF> tempoffsets;
    offsets.base().copyToHost(tempoffsets,level);

    ImageIO::writeImage(tempoffsets, filePath + ".offsets_0.exr", _image_width, _image_height);
}

void WorkingBuffers::clearHistory()
{
    if (!_is_initialized)
        return;

    if (_history_params.size() == 0)
        return;

    // If we are stepping through the history, jump back
    // to the current state before deleting everything.
    if (_history_position > 0)
        gotoHistoryPosition(0);

    // Argh... c++ templates can't be put in a useful list. Must call each of these
    // functions individually.
    offsets.clearHistory();
    residualCache.clearHistory();
    canvasOutputCache.clearHistory();
    canvasVelFCache.clearHistory();
    canvasVelBCache.clearHistory();
    distTransCache.clearHistory();
    offsetsHistogram.clearHistory();

    advectedF.clearHistory();
    advectedB.clearHistory();

    ribbonField.clearHistory();

    _history_params.clear();
    _history_position = 0;
}

void WorkingBuffers::takeHistorySnapshot(const QString& message)
{
    if (!_is_initialized)
        return;

    // Not sure exactly what to do if we take a snapshot
    // while stepping through the history. Make that impossible
    // for now.
    assert(_history_position == 0);

    offsets.copyCurrentToHistory();
    residualCache.copyCurrentToHistory();
    canvasOutputCache.copyCurrentToHistory();
    canvasVelFCache.copyCurrentToHistory();
    canvasVelBCache.copyCurrentToHistory();
    distTransCache.copyCurrentToHistory();
    offsetsHistogram.copyCurrentToHistory();

    advectedF.copyCurrentToHistory();
    advectedB.copyCurrentToHistory();

    ribbonField.copyCurrentToHistory();

    HistoryParams params;
    params.level = _current_level;
    params.message = message;
    _history_params.push_front(params);
    if(_history_params.size() > CudaImageBufferHost<float>::maxSizeHistory){
        _history_params.takeLast();
    }
}

bool WorkingBuffers::gotoHistoryPosition(int position)
{
    assert(_is_initialized);

    if (position < 0 || position >= offsets.historySize())
        return false;

    offsets.copyHistoryToCurrent(position);
    residualCache.copyHistoryToCurrent(position);
    canvasOutputCache.copyHistoryToCurrent(position);
    canvasVelFCache.copyHistoryToCurrent(position);
    canvasVelBCache.copyHistoryToCurrent(position);
    distTransCache.copyHistoryToCurrent(position);
    offsetsHistogram.copyHistoryToCurrent(position);

    advectedF.copyHistoryToCurrent(position);
    advectedB.copyHistoryToCurrent(position);

    ribbonField.copyHistoryToCurrent(position);

    _history_position = position;

    std::cout << "History " << position << ": " << _history_params[_history_position].message.toStdString() << std::endl;
    return true;
}

Color4 WorkingBuffers::outputPixel(int x, int y)
{
    if (!_is_initialized)
        return Color4();
    return canvasOutputCache.pixel(x,y);
}

float4 WorkingBuffers::residualPixel(int x, int y)
{
    if (!_is_initialized)
        return make_float4(-1.f,-1.f,-1.f,-1.f);
    return make_float4(residualCache.base().pixel(x,y), residualCache.working().pixel(x,y),
                       advectedF.residual().pixel(x,y),advectedB.residual().pixel(x,y));
}

PatchXF WorkingBuffers::offsetsPixel(int x, int y)
{
    if (!_is_initialized)
        return PatchXF();
    return offsets.base().pixel(x, y);
}

float WorkingBuffers::histogramPixel(int x, int y)
{
    if (!_is_initialized)
        return 0.0;
    return offsetsHistogram.base().pixel(x, y);
}

int WorkingBuffers::currentLevel() const
{
    if (!_is_initialized)
        return 0;
    int level = _current_level;
    if (_history_position > 0)
        level = _history_params[_history_position].level;

    return level;
}

int WorkingBuffers::currentImageWidth() const
{
    if (!_is_initialized)
        return 0;
    int level = _current_level;
    if (_history_position > 0)
        level = _history_params[_history_position].level;

    return offsets.width(level);
}

int WorkingBuffers::currentImageHeight() const
{
    if (!_is_initialized)
        return 0;
    int level = _current_level;
    if (_history_position > 0)
        level = _history_params[_history_position].level;

    return offsets.height(level);
}

int WorkingBuffers::currentStyleWidth(int styleIndex) const
{
    return DataAccess::instance().getStyleWidth(styleIndex) >> currentLevel();
}

int WorkingBuffers::currentStyleHeight(int styleIndex) const
{
    return DataAccess::instance().getStyleHeight(styleIndex) >> currentLevel();
}

bool WorkingBuffers::copyToGL(struct cudaGraphicsResource* destination, TsOutputType outputType, int level, int styleIndex,
                              float vizGain, bool vizNormalize, int vizMode)
{
    if (!_is_initialized)
        return false;

    cudaArray* d_result;

    cudaError_t error = cudaGraphicsMapResources(1, &destination, 0);
    if (error != cudaSuccess) {std::cerr << "ERROR! GraphicsMapResources" << std::endl; }
    assert(error == cudaSuccess);

    error = cudaGraphicsSubResourceGetMappedArray(&d_result, destination, 0, 0);
    if (error != cudaSuccess) {std::cerr << "ERROR! GraphicsSubResourceGetMappedArray" << std::endl; }
    assert(error == cudaSuccess);

    if (_history_position > 0)
        level = _history_params[_history_position].level;

    copyOutputArray(outputType, level, d_result, styleIndex, vizGain, vizNormalize, vizMode);

    error = cudaGraphicsUnmapResources(1, &destination, 0);
    if (error != cudaSuccess) {std::cerr << "ERROR! GraphicsUnmapResources" << std::endl; }
    assert(error == cudaSuccess);

    return true;
}
