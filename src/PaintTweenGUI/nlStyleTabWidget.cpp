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

#include "nlStyleTabWidget.h"
#include <cassert>
#include <iostream>

#include <cmath>

#include <QFrame>
#include <QMessageBox>
#include <QGridLayout>
#include <QMouseEvent>
#include <QMenuBar>
#include <QPainter>
#include <QPen>

NLStyleTabWidget::NLStyleTabWidget(QWidget* parent) : QTabWidget(parent)
{ 
    _numTabsPerStyle = 1;
    connect(this,SIGNAL(currentChanged(int)),this,SLOT(currentIndexChanged(int)));
}

NLStyleViewerWidget* NLStyleTabWidget::imageViewer(int index) {
    assert(index < imageViewers.size());
    return imageViewers[index];
}

void NLStyleTabWidget::updateIndexFromStyle(int numStyle) {
    if (imageViewers.size() == 0)
        return;

    // multiply by numTabsPerStyle to account for interleaved orientations.
    setCurrentIndex(numStyle*_numTabsPerStyle);
}

void NLStyleTabWidget::drawBox(QPoint image_p, float theta, float radius, int numStyle) {
    if (imageViewers.size() == 0)
        return;

    if(image_p.x() < 0 || image_p.y() < 0 || numStyle < 0 || numStyle*_numTabsPerStyle > (int)imageViewers.size()
//            ||
//            image_p.x() >= (int)imageViewers[numStyle*_numTabsPerStyle]->pixmapWidth() ||
//            image_p.y() >= (int)imageViewers[numStyle*_numTabsPerStyle]->pixmapHeight()
            ){
        imageViewers[this->currentIndex()]->clear();
        return;
    }

    setCurrentIndex(numStyle*_numTabsPerStyle);
    imageViewers[numStyle*_numTabsPerStyle]->drawBox(image_p, theta, radius);
}

void NLStyleTabWidget::addTab(NLStyleViewerWidget* image, QString text){
    imageViewers.push_back(image);
    QTabWidget::addTab(image, text);
}

void NLStyleTabWidget::clear(){
    blockSignals(true);
    qDeleteAll(imageViewers);
    imageViewers.clear();

    QTabWidget::clear();
    blockSignals(false);
}

void NLStyleTabWidget::setCurrentStyle(int index){
    if(this->currentIndex()/_numTabsPerStyle != index)
        setCurrentIndex(index*_numTabsPerStyle);
}

void NLStyleTabWidget::currentIndexChanged(int index){
    if(index%_numTabsPerStyle == 0)
        emit currentStyleChanged((int)floorf((float)index/(float)_numTabsPerStyle));
    else if (index%_numTabsPerStyle == 1)
        emit currentOrientationChanged((int)floorf((float)index/(float)_numTabsPerStyle));
}
