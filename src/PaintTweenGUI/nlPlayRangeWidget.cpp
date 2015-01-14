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

#include "nlPlayRangeWidget.h"
#include "nlSynthesizer.h"

#include <QtGui/QLabel>
#include <QtGui/QSpinBox>
#include <QtGui/QToolButton>
#include <QtGui/QHBoxLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QRadioButton>

#include <iostream>

NLPlayRangeWidget::NLPlayRangeWidget(QWidget* parent, NLSynthesizer* synthesizer)
    : QWidget(parent),
      _lower(1),
      _upper(1),
      _synthesizer(synthesizer)
{
    // Create main layout:
    QHBoxLayout* mainLayout = new QHBoxLayout(this);

    // Create labels:
    _startFrameLabel = new QLabel(tr("Start"), this);
    _endFrameLabel = new QLabel(tr("End"), this);

    // Create spin boxes:
    _startFrameBox = new QSpinBox(this);
    _startFrameBox->setMinimum(_lower);
    _startFrameBox->setMaximum(_upper);
    _curFrameBox = new QSpinBox(this);
    _curFrameBox->setMinimum(_lower);
    _curFrameBox->setMaximum(_upper);
    _endFrameBox = new QSpinBox(this);
    _endFrameBox->setMinimum(_lower);
    _endFrameBox->setMaximum(_upper);

    connect(_startFrameBox, SIGNAL(valueChanged(int)), this, SIGNAL(startFrameChanged(int)));
    connect(_curFrameBox, SIGNAL(valueChanged(int)), this, SIGNAL(curFrameChanged(int)));
    connect(_endFrameBox, SIGNAL(valueChanged(int)), this, SIGNAL(endFrameChanged(int)));

    _curLevelBox = new QSpinBox(this);
    _curLevelBox->setMinimum(0);
    _curLevelBox->setMaximum(10);
    QLabel* curLevelLabel = new QLabel(tr("Level"));
    QLabel* curPassLabel = new QLabel(tr("Pass"));

    _curPassBox = new QSpinBox(this);
    _curPassBox->setMinimum(0);
    _curPassBox->setMaximum(10);

    connect(_curLevelBox, SIGNAL(valueChanged(int)), this, SIGNAL(curLevelChanged(int)));
    connect(_curPassBox, SIGNAL(valueChanged(int)), this, SIGNAL(curPassChanged(int)));

    // Create buttons:
    _backButton = new QToolButton(this);
    _backButton->setIcon(QIcon(":/icons/Back.png"));
    _forwardButton = new QToolButton(this);
    _forwardButton->setIcon(QIcon(":/icons/Forward.png"));

    connect(_backButton, SIGNAL(clicked()), this, SLOT(backward()));
    connect(_forwardButton, SIGNAL(clicked()), this, SLOT(forward()));

    // Create synthesis mode box
    _modeBox = new QGroupBox(tr(""));
    _modeStillButton = new QRadioButton(tr("Still"));
    _modeAnimationButton = new QRadioButton(tr("Animation"));
    _modeStillButton->setChecked(true);
    QHBoxLayout* modeLayout = new QHBoxLayout();
    modeLayout->addWidget(_modeStillButton);
    modeLayout->addWidget(_modeAnimationButton);
    _modeBox->setLayout(modeLayout);

    connect(_modeStillButton, SIGNAL(clicked()), this, SLOT(modeStill()));
    connect(_modeAnimationButton, SIGNAL(clicked()), this, SLOT(modeAnimation()));

    mainLayout->addWidget(_startFrameLabel);
    mainLayout->addWidget(_startFrameBox);
    mainLayout->addWidget(_backButton);
    mainLayout->addWidget(_curFrameBox);
    mainLayout->addWidget(_forwardButton);
    mainLayout->addWidget(_endFrameLabel);
    mainLayout->addWidget(_endFrameBox);
    mainLayout->addWidget(curLevelLabel);
    mainLayout->addWidget(_curLevelBox);
    mainLayout->addWidget(curPassLabel);
    mainLayout->addWidget(_curPassBox);
    mainLayout->addWidget(_modeBox);

    setLayout(mainLayout);
}

void NLPlayRangeWidget::updateRangeFromSynthesizer(){
    // Block signals here so that changing control
    // values don't rebound and set the synthesizer.
    blockSignals(true);

    int firstFrame = _synthesizer->firstFrame();
    int lastFrame = _synthesizer->lastFrame();

    if(lastFrame < firstFrame){
        std::cerr<<"WARNING! NLPlayRangeWidget.cpp: first frame exceeds last frame.\n";
        std::cerr<<"Setting last frame = first frame.\n";
        lastFrame = firstFrame;
    }

    setLower(firstFrame);
    setUpper(lastFrame);

    if (firstFrame != _startFrameBox->value()){
        if (firstFrame >= _lower){
            _startFrameBox->setValue(firstFrame);
        }
    }

    if (lastFrame != _endFrameBox->value()){
        if (lastFrame <= _upper){
            _endFrameBox->setValue(lastFrame);
        }
    }

    _curFrameBox->setValue(_synthesizer->getCurrentPreviewFrame());

    blockSignals(false);
}

void NLPlayRangeWidget::setLower(int frame){
    // Reset the lower bound of the range:
    _startFrameBox->setMinimum(frame);
    _curFrameBox->setMinimum(frame);
    _endFrameBox->setMinimum(frame);

    _lower = frame;
}

void NLPlayRangeWidget::setUpper(int frame){
    // Reset the upper bound of the range:
    _startFrameBox->setMaximum(frame);
    _curFrameBox->setMaximum(frame);
    _endFrameBox->setMaximum(frame);

    _upper = frame;
}

void NLPlayRangeWidget::backward(){
    _curFrameBox->setValue(_curFrameBox->value() - 1);
}

void NLPlayRangeWidget::forward(){
    _curFrameBox->setValue(_curFrameBox->value() + 1);
}

void NLPlayRangeWidget::modeStill()
{
    emit synthesisModeChanged(0);
}
void NLPlayRangeWidget::modeAnimation()
{
    emit synthesisModeChanged(1);
}

void NLPlayRangeWidget::update()
{
    blockSignals(true);
    _curFrameBox->setValue(_synthesizer->getCurrentPreviewFrame());
    _curLevelBox->setValue(_synthesizer->getCurrentPreviewLevel());
    _curPassBox->setValue(_synthesizer->getCurrentPreviewPass());
    blockSignals(false);
}

void NLPlayRangeWidget::bringBackToRange(){
    if (_startFrameBox->value() < _lower){
        _startFrameBox->setValue(_lower);
    }

    if (_curFrameBox->value() < _lower){
        _curFrameBox->setValue(_lower);
    }

    if (_curFrameBox->value() > _upper){
        _curFrameBox->setValue(_upper);
    }

    if (_endFrameBox->value() > _upper){
        _endFrameBox->setValue(_upper);
    }
}
