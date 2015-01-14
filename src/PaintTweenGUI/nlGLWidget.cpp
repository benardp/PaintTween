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

#include "nlGLWidget.h"

#include <QtCore/QDebug>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "PaintTweenCUDA/cudaHostUtil.h"
#include "nlSynthesizer.h"

NLGLWidget::NLGLWidget(QWidget *parent, NLSynthesizer* synthesizer)
    : QGLWidget(parent), _synthesizer(synthesizer)
{
    setFocusPolicy(Qt::StrongFocus);
    _graphicsResource = 0;
    _gl_width = 0; _gl_height = 0;
    _img_width = 0; _img_height = 0;
    _texture = -1;
    _checkboardTexture = -1;
    _use_Checkboard = true;
    _use_sRgb = false;

    connect(_synthesizer, SIGNAL(synthesisAdvanced()), this, SLOT(update()));
    connect(_synthesizer, SIGNAL(cleaningUp()), this, SLOT(cleanupTexture()));
}

void NLGLWidget::changeColorSpace(int sRgb)
{
    _use_sRgb = (sRgb == 1);
    update();
}

void NLGLWidget::changeBackground(int back)
{
    _use_Checkboard = (back == 0);
    update();
}

void NLGLWidget::initializeGL()
{
    QColor color = palette().window().color();
    glClearColor(color.redF(),color.greenF(),color.blueF(),1.f);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
}

void NLGLWidget::resizeGL(int w, int h)
{
    _gl_width = w;
    _gl_height = h;
}

void NLGLWidget::initializeCheckboard()
{
    QImage pattern(":/icons/Checkerboard.png");
    pattern = QGLWidget::convertToGLFormat(pattern);
    GLuint checkboardtex;
    glGenTextures(1, &checkboardtex);
    _checkboardTexture = checkboardtex;
    _checkboard_width = pattern.width();
    _checkboard_height = pattern.height();
    glBindTexture(GL_TEXTURE_2D, _checkboardTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _checkboard_width, _checkboard_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pattern.bits());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindTexture(GL_TEXTURE_2D, 0);
}


void NLGLWidget::initializeTexture()
{
    if (_texture >= 0) {
        cleanupTexture();
    }

    // Create the texture for displaying the result:
    GLuint gltex;
    glGenTextures(1, &gltex);
    _texture = gltex;
    reportGLError("setupGLRendering() genTexture");

    // Bind texture:
    glBindTexture(GL_TEXTURE_2D, _texture);
    reportGLError("setupGLRendering() bindTexture");

    // Allocate texture:
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, _img_width, _img_height, 0, GL_RGBA, GL_FLOAT, NULL);
    reportGLError("setupGLRendering() texImage2D");

    // Set texture parameters:
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    reportGLError("setupGLRendering() glTexParameteri");

    // Unbind texture:
    glBindTexture(GL_TEXTURE_2D, 0);
    reportGLError("setupGLRendering() bindTexture(0)");

    // Register the buffer object:
    checkCUDAError("Pre gl register");
    cudaGraphicsGLRegisterImage(&_graphicsResource, _texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
    checkCUDAError("Post gl register");
}

void NLGLWidget::cleanupTexture()
{
    if (_texture < 0)
        return;

    // Unregister the graphics resource:
    cudaError_t error = cudaGraphicsUnregisterResource(_graphicsResource);
    if (error != cudaSuccess) {
        qCritical() << "ERROR! (cleanupGLRendering()) GraphicsUnregisterResource" << cudaGetErrorString(error);
    }

    // Delete texture:
    GLuint gltex = _texture;
    glDeleteTextures(1, &gltex);
    _texture = -1;

    if(_checkboardTexture < 0)
        return;

    gltex = _checkboardTexture;
    glDeleteTextures(1, &gltex);
    _checkboardTexture = -1;
    reportGLError("OutputTexture cleanup");
}
