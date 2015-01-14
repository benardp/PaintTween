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

#ifndef STYLE_H
#define STYLE_H

#include "types.h"
#include "imagePyramid.h"
#include "dataAccess.h"

#include <QtCore/QVector>

class Style
{

public:
    Style();

    bool load(int level);
    void clear();

    void initializeKeyToKeyRibbon(const QVector<RibbonP>& data, int frame, int width, int height, bool timeIsForwards, bool writeToOutput);
    bool loadKeyToKeyRibbon(int frame, bool timeIsForwards);

    int firstKeyFrameFullIndex() const { return DataAccess::instance().keyFrameIndices().indexOf(_keyFrameSubrange.first()); }
    int lastKeyFrameFullIndex() const { return DataAccess::instance().keyFrameIndices().indexOf(_keyFrameSubrange.last()); }
    int keyFrameFullIndex(int frame) const { return DataAccess::instance().keyFrameIndices().indexOf(frame); }

    const QVector<int>& keyFrameSubrange() const { return _keyFrameSubrange; }
    int keyFrameSubrangeIndex(int frame) const;
    void updateKeyFrameSubrange(int firstFrame, int lastFrame);

    float keyFrameAdvectWeight(int frame, int firstFrame, int lastFrame, bool timeIsForwards) const;

    const ImagePyramid<Color4>& base(int layer) const { return _exemplarBase.at(layer); }
    const ImagePyramid<Color4>& output(int layer) const { return _exemplarOutput.at(layer); }
    const ImagePyramid<float>& orientation(int layer) const { return _exemplarOrientation.at(layer); }
    const ImagePyramid<float>& distanceTransform(int layer) const { return _exemplarDistanceTransform.at(layer); }
    const ImagePyramid<int>& objectIDs(int layer) const { return _exemplarObjectIDs.at(layer); }

protected:
    // Exemplar-sized images
    QVector< ImagePyramid<Color4> > _exemplarOutput;
    QVector< ImagePyramid<Color4> > _exemplarBase;
    QVector< ImagePyramid<float> >  _exemplarOrientation;
    QVector< ImagePyramid<float> >  _exemplarDistanceTransform;
    QVector< ImagePyramid<int> >    _exemplarObjectIDs;

    // Input-sized images
    QVector< ImagePyramid<Color4> > _keyToKeyRibbonB;
    QVector< ImagePyramid<Color4> > _keyToKeyRibbonF;

    QVector<int> _keyFrameSubrange;

    int _styleWidth, _styleHeight;
};

#endif
