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

#include "nlImageViewerWidget.h"

#include <cmath>

#include <QtGui/QFrame>
#include <QtGui/QStyleOption>
#include <QtGui/QMessageBox>
#include <QtGui/QGridLayout>
#include <QtGui/QMenuBar>
#include <QtGui/QPainter>
#include <QtGui/QPen>

#include "PaintTweenCUDA/cudaHostUtil.h"

NLStyleViewerWidget::NLStyleViewerWidget(QWidget* parent, NLSynthesizer* synthesizer, int styleNum, StyleElements type)
    : NLGLWidget(parent,synthesizer)
{
    _showVizBox = false;
    _styleNum = styleNum;
    _type = type;
}

void NLStyleViewerWidget::drawBox(QPoint image_p, float theta, float radius)
{
    _vizPoint = image_p;
    _vizTheta = theta;
    _vizRadius = radius;
    _showVizBox = true;
    update();
}

void NLStyleViewerWidget::clear()
{
    _showVizBox = false;
    update();
}

void NLStyleViewerWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);
    if(isHidden() || !_synthesizer->workingBuffers()->isInitialized())
        return;

    int baseWidth = _synthesizer->getStyleWidth(_styleNum);
    int baseHeight =  _synthesizer->getStyleHeight(_styleNum);

    float imgWidth = _synthesizer->currentStyleWidth(_styleNum);
    float imgHeight = _synthesizer->currentStyleHeight(_styleNum);

    bool need_init = _texture == -1 || imgWidth != _img_width || imgHeight != _img_height;

     if (need_init) {
         _img_width = imgWidth;
         _img_height = imgHeight;
         initializeTexture();
     }

     if (!_synthesizer->copyToGL(_graphicsResource, _styleNum, _type))
         return;

     float wWidth = std::min<float>(_gl_width,baseWidth); // window width & height
     float wHeight = std::min<float>(_gl_height,baseHeight);

     _scaled_width = wWidth;
     _scaled_height = wHeight;
     if ( (wWidth/wHeight) > (imgWidth/imgHeight) ) { // window wider than texture
         _scaled_width = (int) (wHeight * (imgWidth/imgHeight));
     }
     else { // window is narrower than texture
         _scaled_height = (int) (wWidth * (imgHeight/imgWidth));
     }

     float sBound = 1;
     float tBound = 1;

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

    glClear(GL_COLOR_BUFFER_BIT);

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
    }else{
        glClearColor(0.0,0.0,0.0,1.0);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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

        QPointF b0 = QPointF(cos(_vizTheta)*_vizRadius, sin(_vizTheta)*_vizRadius);
        QPointF b1 = QPointF(-sin(_vizTheta)*_vizRadius, cos(_vizTheta)*_vizRadius);
        QPointF center = QPointF(_vizPoint.x(), _vizPoint.y());

        glBegin(GL_LINES);
        glColor4f(0.0,1.0,0.0,1.0);
        glVertex2f(center.x(),center.y()); glVertex2f(center.x() + b1.x(),center.y() + b1.y());;
        glColor4f(1.0,0.0,0.0,1.0);
        glVertex2f(center.x(),center.y()); glVertex2f(center.x() + b0.x(),center.y() + b0.y());
        glVertex2f(center.x() + b0.x() + b1.x(),center.y() + b0.y() + b1.y()); glVertex2f(center.x() - b0.x() + b1.x(),center.y() - b0.y() + b1.y());
        glVertex2f(center.x() - b0.x() + b1.x(),center.y() - b0.y() + b1.y()); glVertex2f(center.x() - b0.x() - b1.x(),center.y() - b0.y() - b1.y());
        glVertex2f(center.x() - b0.x() - b1.x(),center.y() - b0.y() - b1.y()); glVertex2f(center.x() + b0.x() - b1.x(),center.y() + b0.y() - b1.y());
        glVertex2f(center.x() + b0.x() - b1.x(),center.y() + b0.y() - b1.y()); glVertex2f(center.x() + b0.x() + b1.x(),center.y() + b0.y() + b1.y());
        glEnd();

        glColor4f(1.0,1.0,1.0,1.0);
    }

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
}

NLImageViewerWidget::NLImageViewerWidget(QWidget *parent, NLSynthesizer *synthesizer, InputElements type)
    : NLGLWidget(parent,synthesizer), _type(type)
{
}

void NLImageViewerWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);
    if(isHidden() || !_synthesizer->workingBuffers()->isInitialized())
        return;

    int baseWidth, baseHeight;
    _synthesizer->getImageDimensions(baseWidth, baseHeight);
    float imgWidth = _synthesizer->workingBuffers()->currentImageWidth();
    float imgHeight = _synthesizer->workingBuffers()->currentImageHeight();

    bool need_init = _texture == -1 || imgWidth != _img_width || imgHeight != _img_height;

     if (need_init) {
         _img_width = imgWidth;
         _img_height = imgHeight;
         initializeTexture();
     }

     if (!_synthesizer->copyToGL(_graphicsResource, _type))
         return;

     float wWidth = std::min<float>(_gl_width,baseWidth); // window width & height
     float wHeight = std::min<float>(_gl_height,baseHeight);

     _scaled_width = wWidth;
     _scaled_height = wHeight;
     if ( (wWidth/wHeight) > (imgWidth/imgHeight) ) { // window wider than texture
         _scaled_width = (int) (wHeight * (imgWidth/imgHeight));
     }
     else { // window is narrower than texture
         _scaled_height = (int) (wWidth * (imgHeight/imgWidth));
     }

     float sBound = 1;
     float tBound = 1;

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

    glClear(GL_COLOR_BUFFER_BIT);

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

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
}
