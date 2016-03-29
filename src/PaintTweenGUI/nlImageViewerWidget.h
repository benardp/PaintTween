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

#ifndef NLIMAGEVIEWERWIDGET_H
#define NLIMAGEVIEWERWIDGET_H

#include <QMouseEvent>

#include "nlGLWidget.h"

class NLStyleViewerWidget : public NLGLWidget
{
    Q_OBJECT

public:
    NLStyleViewerWidget(QWidget* parent, NLSynthesizer *synthesizer, int styleNum, StyleElements type);

    virtual QSize sizeHint () const { return QSize(560,315); }

public slots:
    void clear();
    void drawBox(QPoint p, float theta, float radius);

signals:
    void clicked(int x, int y);

protected:
    void paintGL();

    // Visualizations
    QPoint _vizPoint;
    float _vizTheta, _vizRadius;
    bool _showVizBox;

private:
    int _styleNum;
    StyleElements _type;
};

class NLImageViewerWidget : public NLGLWidget
{
    Q_OBJECT

public:
    NLImageViewerWidget(QWidget* parent, NLSynthesizer *synthesizer, InputElements type);

protected:
    void paintGL();

private:
    InputElements _type;
};

#endif // NLIMAGEVIEWERWIDGET_H
