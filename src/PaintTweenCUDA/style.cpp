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

#include "style.h"

#include <iostream>

#include <QtCore/QDebug>

#include "texSynth_kernel.h"
#include "imageIO.h"
#include "imagePyramid.h"
#include "stats.h"

using namespace std;

Style::Style()
{
    clear();
}

void Style::clear()
{
    _exemplarBase.clear();
    _exemplarOutput.clear();
    _exemplarOrientation.clear();
    _exemplarDistanceTransform.clear();
    _exemplarObjectIDs.clear();

    _keyToKeyRibbonB.clear();
    _keyToKeyRibbonF.clear();

    _keyFrameSubrange.clear();

    _styleWidth = 0; _styleHeight = 0;
}

bool Style::load(int level)
{
    __TIME_CODE_BLOCK("Style::load");

    DataAccess& data = DataAccess::instance();
    int numStyles = data.getNumStyles();
    assert(numStyles > 0);

    if(_exemplarBase.isEmpty()){

        cout << "PaintTween: Load style..." << flush;

        _exemplarOutput.resize(numStyles);
        _exemplarBase.resize(numStyles);
        _exemplarOrientation.resize(numStyles);
        _exemplarDistanceTransform.resize(numStyles);
        _exemplarObjectIDs.resize(numStyles);

        data.getMaxStyleDimensions(_styleWidth, _styleHeight);
        data.padStyles(_styleWidth,_styleHeight);

        for(int styleIdx=0; styleIdx< numStyles; styleIdx++) {
            // Initialize style size
            data.getStyle(STYLE_INPUT, styleIdx, _exemplarBase[styleIdx]);
            data.getStyle(STYLE_OUTPUT, styleIdx, _exemplarOutput[styleIdx]);
            data.getStyle(STYLE_ORIENTATION, styleIdx, _exemplarOrientation[styleIdx]);
            data.getStyle(STYLE_DIST_TRANS, styleIdx, _exemplarDistanceTransform[styleIdx]);
            data.getStyle(STYLE_OBJECT_ID, styleIdx, _exemplarObjectIDs[styleIdx]);
        }
        qDebug()<<" done.";
    }

    TexSynth::initialize_exemplars(_exemplarBase.first().width(level), _exemplarBase.first().height(level), numStyles);
    _styleWidth = _exemplarBase.first().width(level);
    _styleHeight = _exemplarBase.first().height(level);

    for(int styleIdx=0; styleIdx< numStyles; styleIdx++) {
        if(!_exemplarBase.at(styleIdx).isLoaded(level)){
            _exemplarBase[styleIdx].load(level);
        }
        if(!_exemplarOutput.at(styleIdx).isLoaded(level)){
            _exemplarOutput[styleIdx].load(level);
        }
        if(!_exemplarOrientation.at(styleIdx).isLoaded(level)){
            _exemplarOrientation[styleIdx].load(level);
        }
        if(!_exemplarDistanceTransform.at(styleIdx).isLoaded(level)){
            _exemplarDistanceTransform[styleIdx].load(level);
        }
        if(!_exemplarObjectIDs.at(styleIdx).isLoaded(level)){
            _exemplarObjectIDs[styleIdx].load(level);
        }

        TexSynth::uploadImage_Color4(level, TS_LAYER_EXEMPLAR_BASE, _exemplarBase.at(styleIdx), styleIdx);
        TexSynth::uploadImage_Color4(level, TS_LAYER_EXEMPLAR_OUTPUT, _exemplarOutput.at(styleIdx), styleIdx);
        TexSynth::uploadImage_float(level, TS_LAYER_EXEMPLAR_ORIENTATION, _exemplarOrientation.at(styleIdx), styleIdx);
        TexSynth::uploadImage_float(level, TS_LAYER_EXEMPLAR_DIST_TRANS, _exemplarDistanceTransform.at(styleIdx), styleIdx);
        TexSynth::uploadImage_int(level, TS_LAYER_EXEMPLAR_OBJECT_IDS, _exemplarObjectIDs.at(styleIdx), styleIdx);
    }

    if(_keyToKeyRibbonB.size() > 0 && _keyToKeyRibbonB[0].isInitialized()){
        int fullIdx = keyFrameFullIndex(_keyFrameSubrange.first());
        TexSynth::initialize_cumulativeFields(_keyToKeyRibbonB[fullIdx].width(level), _keyToKeyRibbonB[fullIdx].height(level), _keyFrameSubrange.size());
        for(int i=0; i<_keyFrameSubrange.size(); i++){
            fullIdx = keyFrameFullIndex(_keyFrameSubrange[i]);
            if (!_keyToKeyRibbonB[fullIdx].isLoaded(level)) {
                _keyToKeyRibbonB[fullIdx].load(level);
                _keyToKeyRibbonF[fullIdx].load(level);
            }
            TexSynth::uploadImage_Color4(level, TS_LAYER_CUMUL_FIELD_B, _keyToKeyRibbonB[fullIdx], i);
            TexSynth::uploadImage_Color4(level, TS_LAYER_CUMUL_FIELD_F, _keyToKeyRibbonF[fullIdx], i);
        }
    } else {
        _keyToKeyRibbonF.resize(numStyles);
        _keyToKeyRibbonB.resize(numStyles);
    }

    return true;
}


void Style::initializeKeyToKeyRibbon(const QVector<RibbonP>& data, int frame, int width, int height,
                                     bool timeIsForwards, bool writeToOutput)
{
    int index = keyFrameFullIndex(frame);
    if (timeIsForwards) {
        initializeRibbonPyramid(data, _keyToKeyRibbonB[index], width, height,
                                ImageIO::parsePath(DataAccess::instance().getTemporaryDir()+"/keyToKeyRibbonB_%2.%1.exr",frame));
    } else {
        initializeRibbonPyramid(data, _keyToKeyRibbonF[index], width, height,
                                ImageIO::parsePath(DataAccess::instance().getTemporaryDir()+"/keyToKeyRibbonF_%2.%1.exr",frame));
    }

    if (writeToOutput) {
        QString cachename = ImageIO::parsePath(QString("%1/keyToKeyRibbon%2.%3.exr").arg(DataAccess::instance().getOutDir()).arg((timeIsForwards) ? 'B' : 'F'),frame);
        ImageIO::writeImage(data, cachename, width, height);
    }
}


bool Style::loadKeyToKeyRibbon(int frame, bool timeIsForwards)
{
    QVector<RibbonP> ribbon;
    int width, height;
    QString cachename = ImageIO::parsePath(QString("%1/keyToKeyRibbon%2.%3.exr").arg(DataAccess::instance().getOutDir()).arg((timeIsForwards) ? 'B' : 'F'),frame);
    bool success = ImageIO::readImage(cachename, ribbon, width, height);
    if (success) {
        initializeKeyToKeyRibbon(ribbon, frame, width, height, timeIsForwards, false);
    }
    return success;
}


void Style::updateKeyFrameSubrange(int firstFrame, int lastFrame)
{
    DataAccess& data = DataAccess::instance();
    _keyFrameSubrange.clear();
    for(int i=0 ; i<data.keyFrameIndices().size(); i++){
        int idx = data.keyFrameIndices().at(i);
        if(idx >= firstFrame && idx <= lastFrame){
            _keyFrameSubrange << idx;
        }else {
            if(i<data.keyFrameIndices().size()-1){
                int next_idx = data.keyFrameIndices().at(i+1);
                if(next_idx > firstFrame && next_idx <= lastFrame)
                    _keyFrameSubrange << idx;
            }
            if(i>0){
                int prev_idx = data.keyFrameIndices().at(i-1);
                if(prev_idx >= firstFrame && prev_idx < lastFrame)
                    _keyFrameSubrange << idx;
            }
        }
    }
}

int Style::keyFrameSubrangeIndex(int frame) const
{
    for (int i = 0; i < _keyFrameSubrange.size(); i++) {
        if (_keyFrameSubrange[i] == frame)
            return i;
    }
    return -1;
}

float Style::keyFrameAdvectWeight(int frame, int firstFrame, int lastFrame, bool timeIsForwards) const
{
    int prev_key, prev_frame, next_frame;
    for (prev_key = -1; prev_key < _keyFrameSubrange.size()-1 && frame > _keyFrameSubrange[prev_key+1]; prev_key++) {}
    bool has_prev = (prev_key >= 0);
    bool has_next = (prev_key < _keyFrameSubrange.size()-1);
    prev_frame = (has_prev) ? _keyFrameSubrange[prev_key] : firstFrame;
    next_frame = (has_next) ? _keyFrameSubrange[prev_key+1] : lastFrame;

    float weight;
    if (timeIsForwards) {
        weight = (has_next) ? float(next_frame - frame) / float(next_frame - prev_frame) : 1.0;
    } else {
        weight = (has_prev) ? float(frame - prev_frame) / float(next_frame - prev_frame) : 1.0;
    }
    return weight;
}

