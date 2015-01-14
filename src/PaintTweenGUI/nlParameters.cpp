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

#include "nlParameters.h"

#include <QtCore/QDebug>

NLParameters NLParameters::_instance;

QStringList NLParameters::float_names = QStringList() << "coherenceWeight" << "hysteresisWeight"<<"maxDistortion"
                                                      << "inputAnalogyWeight" << "canvasOutputWeight" << "distanceTransformWeight"
                                                      << "edgeBlendingWeight" << "timeDerivativeInputWeight" << "timeDerivativeOutputWeight" << "temporalCoherenceWeight"
                                                      << "styleObjectIdWeight" << "offsetsHistogramSlope" << "offsetsHistogramThreshold"
                                                      << "advectionJitter" << "lumShiftWeight" << "alphaChannelWeight" << "residualWindowScaleU"
                                                      << "residualWindowScaleV";
QStringList NLParameters::bool_names = QStringList() << "orientationEnabled" << "keyFrameInterpolation"
                                                     << "guidedInterpolation" << "lumShiftEnabled" << "transparencyAllowed";
QStringList NLParameters::int_names = QStringList()  << "version" << "iterations" << "iterationsFirstFrame"
                                                     << "firstPassDirection" << "levels"
                                                     << "maxLevel" << "scatterSamplesCount" << "fixupPasses"
                                                     << "residualWindowSize"  << "firstFrame" << "lastFrame"
                                                     << "colorspace" << "synthesisScheme";
NLParameters::NLParameters()
    : _currentStyleIndex(-1)
{
    clear();
}

void NLParameters::clear()
{
    _float_params.clear();
    _int_params.clear();
    _int_params.clear();
    _offsetsHistogramSlopes.clear();
    _offsetsHistogramThresholds.clear();
    loadDefaults();
}

TsParameters NLParameters::getTsDefaultParams()
{
    TsParameters params;
    params.residualWindowSize = 4;
    params.coherenceWeight = 0.0f;
    params.distTransWeight = 0.0f;
    params.lumShiftWeight = 0.0f;
    params.lumShiftEnable = false;
    params.inputAnalogyWeight = 1.0f;
    params.canvasOutputWeight = 1.0f;
    params.orientationEnabled = true;
    for(int i=0; i<NUM_MAX_STYLES; i++){
        params.offsetsHistogramSlope[i] = 0.0f;
        params.offsetsHistogramThreshold[i] = 0.0f;
    }
    params.numStyles = 1;
    params.maxDistortion = 4.0f;
    params.alphaWeight = 1.0f;
    params.transparencyOk = false;
    params.coherenceRatioRange = 2.0f;
    params.coherenceAngleRange = 0.25f;
    params.levels = 4;
    params.scatterSamplesCount = 25;
    params.advectionJitter = 0.0f;
    params.timeDerivativeInputWeight = 1.f;
    params.timeDerivativeOutputWeight = 1.f;
    params.temporalCoherenceWeight = 0.2f;
    params.hysteresisWeight = 0.f;
    params.interpolateKeyFrame = false;
    params.useGuide = false;
    params.distTransCull = 50;
    params.residualWindowScaleU = 1;
    params.residualWindowScaleV = 1;
    params.firstKeyIndex = -1;
    params.lastKeyIndex = -1;
    params.styleObjIdWeight = 0.f;
    params.direction = BIDIRECTIONAL;
    params.colorspace = 0;

    return params;
}

void NLParameters::loadDefaults()
{
    TsParameters tsParam = getTsDefaultParams();
    setFloat("advectionJitter",tsParam.advectionJitter);
    setFloat("alphaChannelWeight", tsParam.alphaWeight);
    setFloat("canvasOutputWeight", tsParam.canvasOutputWeight);
    setFloat("coherenceWeight", tsParam.coherenceWeight);
    setInt("colorspace", tsParam.colorspace);
    setFloat("distanceTransformWeight", tsParam.distTransWeight);
    setFloat("inputAnalogyWeight", tsParam.inputAnalogyWeight);
    setBool("keyFrameInterpolation", tsParam.interpolateKeyFrame);
    setInt("levels", tsParam.levels);
    setBool("lumShiftEnabled", tsParam.lumShiftEnable);
    setFloat("lumShiftWeight", tsParam.lumShiftWeight);
    setBool("orientationEnabled", tsParam.orientationEnabled);
    setFloat("residualWindowScaleU", tsParam.residualWindowScaleU);
    setFloat("residualWindowScaleV", tsParam.residualWindowScaleV);
    setInt("residualWindowSize", tsParam.residualWindowSize);
    setInt("scatterSamplesCount", tsParam.scatterSamplesCount);
    setFloat("timeDerivativeInputWeight", tsParam.timeDerivativeInputWeight);
    setFloat("timeDerivativeOutputWeight", tsParam.timeDerivativeOutputWeight);
    setFloat("temporalCoherenceWeight", tsParam.temporalCoherenceWeight);
    setFloat("hysteresisWeight", tsParam.hysteresisWeight);
    setFloat("maxDistortion", tsParam.maxDistortion);
    setBool("transparencyAllowed", tsParam.transparencyOk);
    setFloat("styleObjectIdWeight", tsParam.styleObjIdWeight);
    setBool("guidedInterpolation", tsParam.useGuide);
    setInt("firstPassDirection",0);
    setInt("synthesisScheme",S_FMBM_RM);
    setInt("fixupPasses",1);
    setInt("maxLevel",0);
    setInt("iterations",3);
    setInt("iterationsFirstFrame",3);
    _offsetsHistogramSlopes.resize(NUM_MAX_STYLES);
    _offsetsHistogramThresholds.resize(NUM_MAX_STYLES);
    for (int i = 0; i < NUM_MAX_STYLES; i++) {
        _offsetsHistogramSlopes[0] = 0.f;
        _offsetsHistogramThresholds[0] = 0.f;
    }
}

void NLParameters::load(const QDomElement &element, int numStyles)
{
    loadDefaults();

    foreach(QString name, float_names){
        QDomElement child = element.firstChildElement(name);
        if(!child.isNull()){
            if(name == "offsetsHistogramSlope"){
                _offsetsHistogramSlopes.resize(numStyles);
                QString val("value%1");
                for(int i=0; i<numStyles; i++){
                    if(child.hasAttribute(val.arg(i)))
                        _offsetsHistogramSlopes[i] = child.attribute(val.arg(i)).toFloat();
                }
            }else if(name == "offsetsHistogramThreshold"){
                _offsetsHistogramThresholds.resize(numStyles);
                QString val("value%1");
                for(int i=0; i<numStyles; i++){
                    if(child.hasAttribute(val.arg(i)))
                        _offsetsHistogramThresholds[i] = child.attribute(val.arg(i)).toFloat();
                }
            }else{
                _float_params.insert(name, child.attribute("value").toFloat());
            }
        }
    }

    foreach(QString name, bool_names){
        QDomElement child = element.firstChildElement(name);
        if(!child.isNull()){
            _bool_params.insert(name, child.attribute("value").toInt()==1);
        }
    }

    foreach(QString name, int_names){
        QDomElement child = element.firstChildElement(name);
        if(!child.isNull()){
            _int_params.insert(name, child.attribute("value").toInt());
        }
    }

    emit parametersChanged();
    if(!_offsetsHistogramSlopes.isEmpty()){
        _currentStyleIndex = 0;
        emit slopeChanged(_offsetsHistogramSlopes.first());
        emit thresholdChanged(_offsetsHistogramThresholds.first());
    }

}

void NLParameters::save(QDomDocument &doc, QDomElement &element) const
{
    QDomElement params = doc.createElement("params");
    element.appendChild(params);

    foreach(QString name, float_names){
        if(_float_params.contains(name)){
            QDomElement child = doc.createElement(name);
            child.setAttribute("value", _float_params[name]);
            params.appendChild(child);
        }
        if(name == "offsetsHistogramSlope" && !_offsetsHistogramSlopes.isEmpty()){
            QDomElement child = doc.createElement(name);
            for(int i=0; i<_offsetsHistogramSlopes.size(); i++)
                child.setAttribute(QString("value%1").arg(i),_offsetsHistogramSlopes.at(i));
            params.appendChild(child);
        }else if(name == "offsetsHistogramThreshold" && !_offsetsHistogramThresholds.isEmpty()){
            QDomElement child = doc.createElement(name);
            for(int i=0; i<_offsetsHistogramThresholds.size(); i++)
                child.setAttribute(QString("value%1").arg(i),_offsetsHistogramThresholds.at(i));
            params.appendChild(child);
        }
    }

    foreach(QString name, bool_names){
        if(_bool_params.contains(name)){
            QDomElement child = doc.createElement(name);
            child.setAttribute("value", _bool_params[name]);
            params.appendChild(child);
        }
    }

    foreach(QString name, int_names){
        if(_int_params.contains(name)){
            QDomElement child = doc.createElement(name);
            child.setAttribute("value", _int_params[name]);
            params.appendChild(child);
        }
    }
}

float NLParameters::getFloat(const QString &name) const
{
    if(_float_params.contains(name))
        return _float_params.value(name);
    qWarning() << "Unknown parameter" << name;
    return 0.f;
}

bool NLParameters::getBool(const QString &name) const
{
    if(_bool_params.contains(name))
        return _bool_params.value(name);
    qWarning() << "Unknown parameter" << name;
    return false;
}

int NLParameters::getInt(const QString &name) const
{
    if(_int_params.contains(name))
        return _int_params.value(name);
    qWarning() << "Unknown parameter" << name;
    return 0;
}

float NLParameters::offsetsHistogramSlope(int index) const
{
    if(index<_offsetsHistogramSlopes.size())
        return _offsetsHistogramSlopes.at(index);
    qWarning() << "Histogram slope index out of bounds "<< index;
    return 0.f;
}

float NLParameters::offsetsHistogramThreshold(int index) const
{
    if(index<_offsetsHistogramThresholds.size())
        return _offsetsHistogramThresholds.at(index);
    qWarning() << "Histogram threshold index out of bounds "<< index;
    return 0.f;
}

void NLParameters::styleChanged(int i)
{
    _currentStyleIndex = i;
    emit slopeChanged(offsetsHistogramSlope(i));
    emit thresholdChanged(offsetsHistogramThreshold(i));
}

void NLParameters::setOffsetsHistogramSlope(double value)
{
    if(_offsetsHistogramSlopes.size() < _currentStyleIndex)
        _offsetsHistogramSlopes.resize(_currentStyleIndex);
    _offsetsHistogramSlopes[_currentStyleIndex] = value;
    emit parametersChanged();
}

void NLParameters::setOffsetsHistogramThreshold(double value)
{
    if(_offsetsHistogramThresholds.size() < _currentStyleIndex)
        _offsetsHistogramThresholds.resize(_currentStyleIndex);
    _offsetsHistogramThresholds[_currentStyleIndex] = value;
    emit parametersChanged();
}

void NLParameters::setFloat(const QString &name, float value)
{
    _float_params[name] = value;
    emit parametersChanged();
}

void NLParameters::setBool(const QString &name, bool value)
{
    _bool_params[name] = value;
    emit parametersChanged();
}

void NLParameters::setInt(const QString &name, int value)
{
    _int_params[name] = value;
    emit parametersChanged();
}

bool NLParameters::hasParam(QString param) const
{
    bool found = false;
    found = found || _float_params.contains(param);
    found = found || _int_params.contains(param);
    found = found || _bool_params.contains(param);
    return found;
}
