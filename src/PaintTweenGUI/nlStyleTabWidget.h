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

#ifndef NLSTYLETABWIDGET
#define NLSTYLETABWIDGET

#include <QtCore/QList>
#include <QtGui/QTabWidget>

#include "nlImageViewerWidget.h"

class NLStyleTabWidget: public QTabWidget
{
    Q_OBJECT

public:
    NLStyleTabWidget(QWidget* parent = 0);
    NLStyleViewerWidget* imageViewer(int index);

    virtual QSize sizeHint () const { return QSize(560,315); }

public slots:
    void drawBox(QPoint image_p, float theta, float radius, int numStyle);
    void updateIndexFromStyle(int numStyle);
    void addTab(NLStyleViewerWidget*, QString text);
    void clear();
    int numTabsPerStyle() { return _numTabsPerStyle; }
    void setNumTabsPerStyle( int v ) { _numTabsPerStyle = v; }
    void setCurrentStyle(int index);
    void currentIndexChanged(int index);

signals:
    void currentStyleChanged(int index);
    void currentOrientationChanged(int index);

private:
    QVector<NLStyleViewerWidget*> imageViewers;
    int _numTabsPerStyle;
};

#endif
