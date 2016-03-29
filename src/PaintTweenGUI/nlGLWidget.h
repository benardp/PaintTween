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

#ifndef NLGLWIDGET_H
#define NLGLWIDGET_H

#ifdef _MSC_VER
	#include "GL/glew.h"
#endif

#include <QGLWidget>

#include "PaintTweenCUDA/texSynth_interface.h"

#include "nlSynthesizer.h"

class NLGLWidget : public QGLWidget
{
    Q_OBJECT

public:
    NLGLWidget(QWidget* parent, NLSynthesizer* synthesizer);

public slots:
    void cleanupTexture();
    void changeColorSpace(int sRgb);
    void changeBackground(int back);

protected:
    void initializeGL();
    void initializeTexture();
    void initializeCheckboard();
    void resizeGL(int width, int height);

protected:
    int _texture;
    int _checkboardTexture;
    int _checkboard_width, _checkboard_height;
    struct cudaGraphicsResource* _graphicsResource;

    float _gl_width, _gl_height;
    int _img_width, _img_height;
    int _scaled_width, _scaled_height;

    bool _use_sRgb;
    bool _use_Checkboard;

    NLSynthesizer* _synthesizer;
};

#endif // NLGLWIDGET_H
