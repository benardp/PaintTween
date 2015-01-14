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

#ifndef SYNTHESISWIDGET_H
#define SYNTHESISWIDGET_H

#include "nlGLWidget.h"

//------------------------------------------------------------------------------
//
// GUI class in charge of displaying the current (still) synthesis to the
// user to give an immediate feedback on changes of style and parameters.
//
//------------------------------------------------------------------------------

class NLSynthesisWidget : public NLGLWidget
{
    Q_OBJECT

public:
    NLSynthesisWidget(QWidget* parent, NLSynthesizer* synthesizer, TsOutputType outputType);
    QWidget* controlWidget() { return _controlWidget; }

    virtual QSize sizeHint () const { return QSize(624,351); }

public slots:
    void removeVizBox() { _showVizBox = false;}
    void vizNormalizeChanged(int state);
    void vizGainChanged(double value);
    void vizModeChanged(int value);

signals:
    void imagePointClicked(int styleIndex);
    void imagePointRefreshed(QPoint p, float theta, float r, int styleIndex);

protected:
    void draw(int imgWidth, int imgHeight, float tBound, float sBound);
    void paintGL();
    void updateBoxVisualization(bool display);

    void mousePressEvent(QMouseEvent* event);

    TsOutputType _outputType;

    QPoint _scaledVizPoint;

    QWidget* _controlWidget;
    bool _vizNormalize;
    double _vizGain;
    int _vizMode;

    static QPoint _clickedPoint;
    static bool _showVizBox;
};

class NLHistogramWidget : public NLSynthesisWidget
{
    Q_OBJECT

public:
    NLHistogramWidget(QWidget* parent, NLSynthesizer* synthesizer);

public slots:
    void styleChanged(int index);

signals:
    void statusMessage(const QString& message);

protected:
    void paintGL();
    void updateBoxVisualization(bool display);

    int _styleIndex;
};


#endif 
