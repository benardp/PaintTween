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

#include "nlDialsAndKnobs.h"

#include <QLayout>
#include <QGroupBox>

NLVariant::NLVariant(QLayout *parent, const QString &paramName, const QString &text)
    : _paramName(paramName)
{
    _label = new QLabel(text);
    connect(&NLParameters::instance(),SIGNAL(parametersChanged()),this,SLOT(update()));
    _layout = new QHBoxLayout;
    _layout->setSpacing(10);
    parent->addItem(_layout);
}

NLFloat::NLFloat(QLayout *parent, const QString &paramName, const QString &text)
    : NLVariant(parent,paramName,text)
{
    _spinBox = new QDoubleSpinBox;
    float value = NLParameters::instance().getFloat(_paramName);
    _spinBox->setValue(value);
    _spinBox->setMinimum(-100);
    _spinBox->setMaximum(100);
    _spinBox->setSingleStep(0.1);
    _spinBox->setDecimals(3);
    connect(_spinBox,SIGNAL(valueChanged(double)),this,SLOT(setValue(double)));

    _layout->addWidget(_label,0,Qt::AlignRight);
    _layout->addWidget(_spinBox);
}

NLFloat::NLFloat(QLayout *parent, const QString &paramName, const QString &text,
                 double lower_limit, double upper_limit, double step_size)
    : NLVariant(parent,paramName,text)
{
    _spinBox = new QDoubleSpinBox;
    float value = NLParameters::instance().getFloat(_paramName);
    _spinBox->setValue(value);
    _spinBox->setMinimum(lower_limit);
    _spinBox->setMaximum(upper_limit);
    _spinBox->setSingleStep(step_size);
    _spinBox->setDecimals(3);
    connect(_spinBox,SIGNAL(valueChanged(double)),this,SLOT(setValue(double)));

    _layout->addWidget(_label,0,Qt::AlignRight);
    _layout->addWidget(_spinBox);
}


void NLFloat::setValue(double f)
{
    float prevValue = NLParameters::instance().getFloat(_paramName);
    if (prevValue != f){
        NLParameters::instance().setFloat(_paramName,f);
        emit valueChanged(f);
    }
}

void NLFloat::update()
{
    _spinBox->setValue(value());
}

float NLFloat::value() const
{
    return NLParameters::instance().getFloat(_paramName);
}


NLBool::NLBool(QLayout *parent, const QString &paramName, const QString &text)
    : NLVariant(parent,paramName,text)
{
    _checkBox = new QCheckBox;
    _checkBox->setChecked(NLParameters::instance().getBool(_paramName));
    connect(_checkBox,SIGNAL(stateChanged(int)),this,SLOT(setValue(int)));

    _layout->addWidget(_label,0,Qt::AlignRight);
    _layout->addWidget(_checkBox);
}

bool NLBool::value() const
{
    return NLParameters::instance().getBool(_paramName);
}

void NLBool::setValue(int s)
{
    bool prevValue = NLParameters::instance().getBool(_paramName);
    bool b = (s == Qt::Checked);
    if (prevValue != b){
        NLParameters::instance().setBool(_paramName,b);
        emit valueChanged(b);
    }
}

void NLBool::update()
{
    _checkBox->setChecked(value());
}


NLInt::NLInt(QLayout *parent, const QString &paramName, const QString &text)
    : NLVariant(parent,paramName,text)
{
    _spinBox = new QSpinBox;
    int value = NLParameters::instance().getInt(_paramName);
    _spinBox->setValue(value);
    _spinBox->setMinimum(-100);
    _spinBox->setMaximum(100);
    _spinBox->setSingleStep(1);
    connect(_spinBox,SIGNAL(valueChanged(int)),this,SLOT(setValue(int)));

    _layout->addWidget(_label,0,Qt::AlignRight);
    _layout->addWidget(_spinBox);
}

NLInt::NLInt(QLayout *parent, const QString &paramName, const QString &text,
             int lower_limit, int upper_limit, int step_size)
    : NLVariant(parent,paramName,text)
{
    _spinBox = new QSpinBox;
    int value = NLParameters::instance().getInt(_paramName);
    _spinBox->setValue(value);
    _spinBox->setMinimum(lower_limit);
    _spinBox->setMaximum(upper_limit);
    _spinBox->setSingleStep(step_size);
    connect(_spinBox,SIGNAL(valueChanged(int)),this,SLOT(setValue(int)));

    _layout->addWidget(_label,0,Qt::AlignRight);
    _layout->addWidget(_spinBox);
}

int NLInt::value() const
{
    return NLParameters::instance().getInt(_paramName);;
}

void NLInt::setValue(int i)
{
    int prevValue = NLParameters::instance().getInt(_paramName);
    if (prevValue != i){
        NLParameters::instance().setInt(_paramName,i);
        emit valueChanged(i);
    }
}

void NLInt::update()
{
    _spinBox->setValue(value());
}

NLStringList::NLStringList(QLayout *parent, const QString &paramName, const QString &text, QStringList &choices)
    : NLVariant(parent,paramName,text)
{
    _comboBox = new QComboBox;
    _comboBox->addItems(choices);
    int value = NLParameters::instance().getInt(_paramName);
    _comboBox->setCurrentIndex(value);
    connect(_comboBox,SIGNAL(currentIndexChanged(int)),this,SLOT(setValue(int)));

    _layout->addWidget(_label,0,Qt::AlignRight);
    _layout->addWidget(_comboBox);
}

int NLStringList::value() const
{
    return NLParameters::instance().getInt(_paramName);
}

void NLStringList::setValue(int i)
{
    int prevValue = NLParameters::instance().getInt(_paramName);
    if (prevValue != i){
        NLParameters::instance().setInt(_paramName,i);
        emit valueChanged(i);
    }
}

void NLStringList::update()
{
    _comboBox->setCurrentIndex(value());
}

NLDialsAndKnobs::NLDialsAndKnobs(QWidget *parent, Qt::WindowFlags flags)
    : QWidget(parent,flags)
{
    QGroupBox* basicParametersGroup = new QGroupBox("Basic Parameters", this);
    QVBoxLayout* basicParametersLayout = new QVBoxLayout();
    _knobs << new NLInt(basicParametersLayout, "residualWindowSize", "Window radius",0,20,1);
    _knobs << new NLFloat(basicParametersLayout, "residualWindowScaleU", "Window scale u",0,20,1);
    _knobs << new NLFloat(basicParametersLayout, "residualWindowScaleV", "Window scale v",0,20,1);
    _knobs << new NLInt(basicParametersLayout, "iterations", "Iterations",1,100,1);
    _knobs << new NLInt(basicParametersLayout, "iterationsFirstFrame", "Iterations on first frame",1,100,1);
    _knobs << new NLInt(basicParametersLayout, "levels", "Levels",1,100,1);
    _knobs << new NLInt(basicParametersLayout, "scatterSamplesCount", "Scattering samples",0,1000,1);
    _knobs << new NLInt(basicParametersLayout, "maxLevel", "Max level",0, 100,1);
    basicParametersGroup->setLayout(basicParametersLayout);

    QGroupBox* residualParametersGroup = new QGroupBox("Residual Parameters", this);
    QVBoxLayout* residualParametersLayout = new QVBoxLayout();
    _knobs << new NLFloat(residualParametersLayout, "coherenceWeight","Coherence Weight",0.0,10.0,0.001);
    _knobs << new NLFloat(residualParametersLayout, "inputAnalogyWeight","Input analogy weight",0.0,10.0,0.01);
    _knobs << new NLFloat(residualParametersLayout, "canvasOutputWeight","Canvas output weight",0.0,10.0,0.01);
    _knobs << new NLFloat(residualParametersLayout, "distanceTransformWeight","Distance transform weight",0.0,10.0,0.01);
    _knobs << new NLFloat(residualParametersLayout, "styleObjectIdWeight","Style object ID weight",0.0,10.0,0.5);
    _knobs << new NLBool(residualParametersLayout, "orientationEnabled", "Use Orientation");
    residualParametersGroup->setLayout(residualParametersLayout);

    QGroupBox* animationParametersGroup = new QGroupBox("Animation Parameters", this);
    QVBoxLayout* animationParametersLayout = new QVBoxLayout();
    QStringList schemeChoices = QStringList() << "F-M-B-M" << "F-M-B-M Rand. Merge" << "F-B-M-O" << "T-C-T-F" << "Independent";
    _knobs << new NLStringList(animationParametersLayout,"synthesisScheme","Synthesis scheme",schemeChoices);
    _knobs << new NLFloat(animationParametersLayout, "timeDerivativeInputWeight","Time deriv. input weight",0.0,10.0,0.01);
    _knobs << new NLFloat(animationParametersLayout, "timeDerivativeOutputWeight","Time deriv. output weight",0.0,10.0,0.01);
    _knobs << new NLFloat(animationParametersLayout, "temporalCoherenceWeight","Temporal coherence weight",0.0,10.0,0.01);
    _knobs << new NLFloat(animationParametersLayout, "hysteresisWeight","Hysteresis weight",0.0,10.0,0.01);
    _knobs << new NLFloat(animationParametersLayout, "maxDistortion","Max distortion",0.0,10.00,0.1);
    _knobs << new NLFloat(animationParametersLayout, "advectionJitter","Advection Jitter",0.0,100.0,1);
    _knobs << new NLInt(animationParametersLayout, "fixupPasses", "Fixup passes",0,100,1);
    QStringList directionsChoices = QStringList() << "forward" << "backward";
    _knobs << new NLStringList(animationParametersLayout,"firstPassDirection","Start advection",directionsChoices);
    QHBoxLayout* tickLayout = new QHBoxLayout();
    NLBool* keyFrameInterpolation = new NLBool(tickLayout,"keyFrameInterpolation","Key-frame interpolation");
    _knobs << keyFrameInterpolation;
    NLBool* guidedInterpolation = new NLBool(tickLayout,"guidedInterpolation","Use guide");
    connect(keyFrameInterpolation->_checkBox, SIGNAL(toggled(bool)), guidedInterpolation->_checkBox, SLOT(setEnabled(bool)));
    _knobs << guidedInterpolation;
    animationParametersLayout->addLayout(tickLayout);
    animationParametersGroup->setLayout(animationParametersLayout);

    QGroupBox* specialParametersGroup = new QGroupBox("Special Parameters", this);
    QVBoxLayout* specialParametersLayout = new QVBoxLayout();
    _knobs << new NLFloat(specialParametersLayout,"alphaChannelWeight","Alpha channel weight",0.0,100.0,0.25);
    _knobs << new NLBool(specialParametersLayout,"transparencyAllowed","Allow transparency in opaque regions");
    QStringList colorspaceChoices = QStringList() << "RGB" << "LAB";
    _knobs << new NLStringList(specialParametersLayout,"colorspace","Colorspace",colorspaceChoices);

    QLabel* offsetsHistogramSlopeLabel = new QLabel("Offsets histogram slope");
    _offsetsHistogramSlope = new QDoubleSpinBox(specialParametersGroup);
    _offsetsHistogramSlope->setRange(0.0, 10.0);
    _offsetsHistogramSlope->setSingleStep(0.01);
    connect(_offsetsHistogramSlope, SIGNAL(valueChanged(double)), &NLParameters::instance(), SLOT(setOffsetsHistogramSlope(double)));
    connect(&NLParameters::instance(), SIGNAL(slopeChanged(double)), _offsetsHistogramSlope, SLOT(setValue(double)));
    QHBoxLayout* blayout = new QHBoxLayout;
    blayout->addWidget(offsetsHistogramSlopeLabel,0,Qt::AlignRight);
    blayout->addWidget(_offsetsHistogramSlope);
    blayout->setSpacing(10);
    specialParametersLayout->addLayout(blayout);

    QLabel* offsetsHistogramThresholdLabel = new QLabel("Offsets histogram threshold");
    _offsetsHistogramThreshold = new QDoubleSpinBox(specialParametersGroup);
    _offsetsHistogramThreshold->setRange(0.0, 1000.0);
    _offsetsHistogramThreshold->setSingleStep(1.0);
    connect(_offsetsHistogramThreshold, SIGNAL(valueChanged(double)), &NLParameters::instance(), SLOT(setOffsetsHistogramThreshold(double)));
    connect(&NLParameters::instance(), SIGNAL(thresholdChanged(double)), _offsetsHistogramThreshold, SLOT(setValue(double)));
    QHBoxLayout* blayout2 = new QHBoxLayout;
    blayout2->addWidget(offsetsHistogramThresholdLabel,0,Qt::AlignRight);
    blayout2->addWidget(_offsetsHistogramThreshold);
    blayout2->setSpacing(10);
    specialParametersLayout->addLayout(blayout2);

    specialParametersGroup->setLayout(specialParametersLayout);

    // Create overall layout and add group boxes:
    QGridLayout* layout = new QGridLayout;
    layout->addWidget(basicParametersGroup, 0, 0, 1, 1);
    layout->addWidget(residualParametersGroup, 1,0,2,1);
    layout->addWidget(animationParametersGroup, 0, 1, 1, 1);
    layout->addWidget(specialParametersGroup, 1, 1, 1, 1);

    setLayout(layout);
}

NLDialsAndKnobs::~NLDialsAndKnobs()
{
    qDeleteAll(_knobs);
}
