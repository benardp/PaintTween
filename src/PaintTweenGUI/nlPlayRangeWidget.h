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

#ifndef NLPLAYRANGEWIDGET_H
#define NLPLAYRANGEWIDGET_H

#include <QtGui/QWidget>

class QLabel;
class QSpinBox;
class QToolButton;
class QGroupBox;
class QRadioButton;

class NLSynthesizer;

class NLPlayRangeWidget : public QWidget
{
    Q_OBJECT

public:
    NLPlayRangeWidget(QWidget* parent, NLSynthesizer* synthesizer);


public slots:
    void updateRangeFromSynthesizer();

    void setLower(int frame);
    void setUpper(int frame);

    void backward();
    void forward();

    void modeStill();
    void modeAnimation();

    void update();

signals:
    void startFrameChanged(int);
    void curFrameChanged(int);
    void endFrameChanged(int);
    void synthesisModeChanged(int);
    void curLevelChanged(int);
    void curPassChanged(int);

protected:
    QLabel* _startFrameLabel;
    QLabel* _endFrameLabel;

    QSpinBox* _startFrameBox;
    QSpinBox* _curFrameBox;
    QSpinBox* _endFrameBox;

    QSpinBox* _curLevelBox;
    QSpinBox* _curPassBox;

    QToolButton* _backButton;
    QToolButton* _forwardButton;

    QGroupBox* _modeBox;
    QRadioButton* _modeStillButton;
    QRadioButton* _modeAnimationButton;

    void bringBackToRange();

    int _lower;
    int _upper;

    NLSynthesizer* _synthesizer;
};

#endif
