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

#include "nlSynthesisWidget.h"
#include "nlSynthesizer.h"
#include <QtGui/QKeyEvent>
#include <QtGui/QHBoxLayout>
#include <QtGui/QCheckBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QSpinBox>
#include <QtGui/QComboBox>
#include <QtGui/QLabel>
#include <QtCore/QDebug>
#include <iostream>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "PaintTweenCUDA/synthesisProcessor.h"
#include "PaintTweenCUDA/cudaHostUtil.h"
#include "PaintTweenCUDA/workingBuffers.h"

bool NLSynthesisWidget::_showVizBox = false;
QPoint NLSynthesisWidget::_clickedPoint(-1,-1);

NLSynthesisWidget::NLSynthesisWidget(QWidget* parent, NLSynthesizer* synthesizer, TsOutputType outputType)
    : NLGLWidget(parent,synthesizer),
      _outputType(outputType)
{
    _controlWidget = new QWidget;
    _vizNormalize = true;
    _vizGain = 1.0;
    _vizMode = 0;

    if (outputType == TS_OUTPUT_RESIDUAL ||
            outputType == TS_OUTPUT_VEL_F ||
            outputType == TS_OUTPUT_VEL_B) {
        QHBoxLayout* layout = new QHBoxLayout;
        QCheckBox* normalize = new QCheckBox("Normalize");
        QDoubleSpinBox* gain = new QDoubleSpinBox();

        gain->setValue(1.0);
        gain->setSingleStep(0.1);
        normalize->setChecked(true);

        layout->addWidget(normalize,1,Qt::AlignRight);
        layout->addWidget(new QLabel("Gain"),0,Qt::AlignRight);
        layout->addWidget(gain,0,Qt::AlignLeft);

        connect(normalize, SIGNAL(stateChanged(int)), this, SLOT(vizNormalizeChanged(int)));
        connect(gain, SIGNAL(valueChanged(double)), this, SLOT(vizGainChanged(double)));
        _controlWidget->setLayout(layout);
    } else if (outputType == TS_OUTPUT_OFFSET ||
               outputType == TS_OUTPUT_ADVECTED_F ||
               outputType == TS_OUTPUT_ADVECTED_B) {
        QHBoxLayout* layout = new QHBoxLayout;
        QComboBox* mode = new QComboBox();

        if (outputType == TS_OUTPUT_OFFSET) {
            QStringList labels = QStringList() << "Offsets" << "Successful Advection" << "Orientation"<<"Grad. X"<<"Grad. Y"<<"Scale";
            mode->addItems(labels);

        } else if (outputType == TS_OUTPUT_ADVECTED_F ||
                   outputType == TS_OUTPUT_ADVECTED_B) {
            QStringList labels = QStringList() << "Colors" << "Offsets" << "Successful Advection" << "Orientation" << "Time Step";
            mode->addItems(labels);
        }

        layout->addWidget(new QLabel("Viz Mode"),1,Qt::AlignRight);
        layout->addWidget(mode,0,Qt::AlignLeft);

        connect(mode, SIGNAL(currentIndexChanged(int)), this, SLOT(vizModeChanged(int)));

        _controlWidget->setLayout(layout);
    }
    _controlWidget->setMinimumHeight(50);
}

void NLSynthesisWidget::vizNormalizeChanged(int state)
{
    _vizNormalize = state; update();
}

void NLSynthesisWidget::vizGainChanged(double value)
{
    _vizGain = value; update();
}

void NLSynthesisWidget::vizModeChanged(int value)
{
    _vizMode = value; update();
}

void NLSynthesisWidget::draw(int imgWidth, int imgHeight, float tBound, float sBound)
{
    // Reshape the GL viewport:
    glViewport(0, _gl_height-_scaled_height, _scaled_width, _scaled_height);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, 1.0, 1.0, 0.0, 0.0, 1.0);

    glDisable(GL_FRAMEBUFFER_SRGB);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);

    if(_use_Checkboard){
        if(_checkboardTexture < 0)
            initializeCheckboard();
        QColor color = palette().window().color();
        glClearColor(color.redF(),color.greenF(),color.blueF(),1.f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, _checkboardTexture);
        renderTexturedQuad((1<<_synthesizer->currentLevel())*imgWidth/(2.f*_checkboard_width),
                           (1<<_synthesizer->currentLevel())*imgHeight/(2.f*_checkboard_height));
        glBindTexture(GL_TEXTURE_2D, 0);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }else{
        glClearColor(0.0,0.0,0.0,1.0);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    if(_use_sRgb)
        glEnable(GL_FRAMEBUFFER_SRGB);

    glBindTexture(GL_TEXTURE_2D, _texture);
    reportGLError("bindTexture");

    // Render result as textured quad:
    renderTexturedQuad(sBound,tBound);

    glBindTexture(GL_TEXTURE_2D, 0);
    reportGLError("bindTexture(0)");

    glDisable(GL_TEXTURE_2D);

    if(_use_sRgb)
        glDisable(GL_FRAMEBUFFER_SRGB);

    if (_showVizBox) {

        glLoadIdentity();
        glOrtho(0.0, imgWidth, imgHeight, 0.0, 0.0, 1.0);
        glLineWidth(2);

        float cx = _scaledVizPoint.x();
        float cy = _scaledVizPoint.y();
        float r = _synthesizer->getIntParameter("residualWindowSize");
        glBegin(GL_LINES);
        glColor4f(0.0,1.0,0.0,1.0);
        glVertex2f(cx, cy); glVertex2f(cx, cy + r);
        glColor4f(1.0,0.0,0.0,1.0);
        glVertex2f(cx, cy); glVertex2f(cx + r, cy);
        glVertex2f(cx + r, cy + r); glVertex2f(cx - r, cy + r);
        glVertex2f(cx - r, cy + r); glVertex2f(cx - r, cy - r);
        glVertex2f(cx - r, cy - r); glVertex2f(cx + r, cy - r);
        glVertex2f(cx + r, cy - r); glVertex2f(cx + r, cy + r);
        glEnd();

        glColor4f(1.0,1.0,1.0,1.0);
    }

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
}

void NLSynthesisWidget::paintGL()
{
    if(isHidden())
        return;

    glClear(GL_COLOR_BUFFER_BIT);

    int baseWidth, baseHeight;
    _synthesizer->getImageDimensions(baseWidth, baseHeight);
    bool need_init = _texture == -1 || baseWidth != _img_width || baseHeight != _img_height;

    if (need_init && _synthesizer->workingBuffers()->isInitialized()) {
        _img_width = baseWidth;
        _img_height = baseHeight;
        initializeTexture();
    }

    if (!_synthesizer->workingBuffers()->copyToGL(_graphicsResource, _outputType, _synthesizer->currentLevel(), 0,
                                                  _vizGain, _vizNormalize, _vizMode))
        return;

    float wWidth = std::min<float>(_gl_width,baseWidth); // window width & height
    float wHeight = std::min<float>(_gl_height,baseHeight);
    float imgWidth = _synthesizer->workingBuffers()->currentImageWidth();
    float imgHeight = _synthesizer->workingBuffers()->currentImageHeight();
    _scaled_width = wWidth;
    _scaled_height = wHeight;

    // Set bounds so texture will be as large as possible w/uniform scaling.

    if ( (wWidth/wHeight) > (imgWidth/imgHeight) ) { // window wider than texture
        _scaled_width = (int) (wHeight * (imgWidth/imgHeight));
    }
    else { // window is narrower than texture
        _scaled_height = (int) (wWidth * (imgHeight/imgWidth));
    }

    float sBound = imgWidth / float(baseWidth);
    float tBound = imgHeight / float(baseHeight);

    updateBoxVisualization(false);

    draw(imgWidth,imgHeight,sBound,tBound);
}

void NLSynthesisWidget::updateBoxVisualization(bool display)
{
    int x = _clickedPoint.x();
    int y = _clickedPoint.y();
    if(x<0 || y<0 || x >= _scaled_width || y >= _scaled_height){
        emit imagePointRefreshed(QPoint(-1,-1), 0, 0, 0);
        _showVizBox = false;
        return;
    }
    float scalex = (float)(_scaled_width) / (float)(_synthesizer->workingBuffers()->currentImageWidth());
    float scaley = (float)(_scaled_height) / (float)(_synthesizer->workingBuffers()->currentImageHeight());
    int ix = floor(x / scalex);
    int iy = floor(y / scaley);
    Color4 output = _synthesizer->workingBuffers()->outputPixel(ix,iy);
    PatchXF offsets = _synthesizer->workingBuffers()->offsetsPixel(ix,iy);
    PatchXF offset_right = _synthesizer->workingBuffers()->offsetsPixel(ix+1,iy);
    PatchXF offset_up = _synthesizer->workingBuffers()->offsetsPixel(ix,iy+1);
    float4 residual = _synthesizer->workingBuffers()->residualPixel(ix,iy);

    if(display) {
        qDebug("mouse pos: %d %d (%d %d)", x, y, ix, iy);
        qDebug("output color: %f %f %f %f", output.r, output.g, output.b, output.a);
        qDebug("offset: %f %f %f %f %d %f x %f",
               offsets.x, offsets.y, offsets.theta, offsets.hysteresis, offsets.layer, offsets.scaleU, offsets.scaleV);
        qDebug("offset diff: do/dx (%f %f) do/dy (%f %f)",
               offset_right.x - offsets.x, offset_right.y - offsets.y,
               offset_up.x - offsets.x, offset_up.y - offsets.y);
        qDebug("residual: %f W: %f F: %f B: %f", residual.x, residual.y, residual.z, residual.w);
    }

    float orientation = offsets.theta;
    float scaled_neighborhood_radius = (float)_synthesizer->getIntParameter("residualWindowSize");

    QPoint offset_p(offsets.x,offsets.y);
    if(display) {
        emit imagePointClicked(offsets.layer);
    }
    emit imagePointRefreshed(offset_p, orientation, scaled_neighborhood_radius, offsets.layer);

    _scaledVizPoint = QPoint(ix,iy);
}

void NLSynthesisWidget::mousePressEvent(QMouseEvent* event)
{
    _clickedPoint = QPoint(event->x(), event->y());
    _showVizBox = true;
    updateBoxVisualization(true);

    update();
}


// Histogram widget

NLHistogramWidget::NLHistogramWidget(QWidget *parent, NLSynthesizer *synthesizer)
    : NLSynthesisWidget(parent,synthesizer,TS_OUTPUT_HISTOGRAM)
{
    _styleIndex = 0;
    _controlWidget = new QWidget;
    _controlWidget->setMinimumHeight(50);
}

void NLHistogramWidget::styleChanged(int index)
{
    if(index>=0){
        _styleIndex = index;
        update();
    }
}

void NLHistogramWidget::paintGL()
{
    if(isHidden())
        return;

    glClear(GL_COLOR_BUFFER_BIT);

    int baseWidth, baseHeight;
    _synthesizer->getMaxStyleDimensions(baseWidth, baseHeight);
    if(baseWidth==0 || baseHeight==0)
        return;

    bool need_init = _texture == -1 || baseWidth != _img_width || baseHeight != _img_height;

    if (need_init && _synthesizer->workingBuffers()->isInitialized()) {
        _synthesizer->getMaxStyleDimensions(_img_width,_img_height);
        initializeTexture();
    }

    _synthesizer->workingBuffers()->copyToGL(_graphicsResource, TS_OUTPUT_HISTOGRAM, _synthesizer->currentLevel(), _styleIndex);

    float wWidth = _gl_width; // window width & height
    float wHeight = _gl_height;
    float imgWidth = _synthesizer->currentStyleWidth(_styleIndex);
    float imgHeight = _synthesizer->currentStyleHeight(_styleIndex);
    /*float exWidth = _synthesizer->getStyleWidth(_styleIndex);
    float exHeight = _synthesizer->getStyleHeight(_styleIndex);*/
    _scaled_width = wWidth;
    _scaled_height = wHeight;

    // Set bounds so texture will be as large as possible w/uniform scaling.

    if ( (wWidth/wHeight) > (imgWidth/imgHeight) ) { // window wider than texture
        _scaled_width = (int) (wHeight * (imgWidth/imgHeight));
    }
    else { // window is narrower than texture
        _scaled_height = (int) (wWidth * (imgHeight/imgWidth));
    }

    float tBound = 1.f / float(1 << _synthesizer->currentLevel());// * imgHeight / exHeight;
    float sBound = 1.f / float(1 << _synthesizer->currentLevel());// * imgWidth / exWidth;

    updateBoxVisualization(false);
    draw(imgWidth,imgHeight,tBound,sBound);
}

void NLHistogramWidget::updateBoxVisualization(bool display)
{
    int x = _clickedPoint.x();
    int y = _clickedPoint.y();

    if(x<0 || y<0 || x >= _scaled_width || y >= _scaled_height){
        emit imagePointRefreshed(QPoint(-1,-1), 0, 0, 0);
        _showVizBox = false;
        return;
    }

    float wStyle = _synthesizer->currentStyleWidth(_styleIndex);
    float hStyle = _synthesizer->currentStyleHeight(_styleIndex);

    float scalex = (float)(_scaled_width) / wStyle;
    float scaley = (float)(_scaled_height) / hStyle;
    int ix = floor(x / scalex);
    int iy = floor(y / scaley);
    float hist = _synthesizer->workingBuffers()->histogramPixel(ix, iy);

    if (display) {
        qDebug("mouse pos: %d %d (%d %d)", x, y, ix, iy);
        qDebug("histogram value: %f", hist);
    }
    QString msg = QString("histogram value: %1").arg(hist);
    emit statusMessage(msg);

    PatchXF offsets(ix,iy,0,4);

    float orientation = offsets.theta;
    float scaled_neighborhood_radius = (float)_synthesizer->getIntParameter("residualWindowSize");

    QPoint offset_p(offsets.x,offsets.y);
    if (display) {
        emit imagePointClicked(_styleIndex);
    }
    emit imagePointRefreshed(offset_p, orientation, scaled_neighborhood_radius, _styleIndex);

    _scaledVizPoint = QPoint(ix,iy);
}
