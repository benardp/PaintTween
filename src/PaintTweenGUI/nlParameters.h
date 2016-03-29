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

#ifndef NLPARAMETERS_H
#define NLPARAMETERS_H

#include <QObject>
#include <QDomDocument>
#include <QHash>
#include <QStringList>
#include <QVector>

#include "PaintTweenCUDA/texSynth_interface.h"

class NLDataAccess;

class NLParameters : public QObject
{
    Q_OBJECT
    friend class NLDataAccess;

public:
    NLParameters();

    void clear();
    void loadDefaults();
    TsParameters getTsDefaultParams();

    void load(const QDomElement &element, int numStyles);
    void save(QDomDocument& doc, QDomElement& element) const;

    float getFloat(const QString& name) const;
    bool  getBool(const QString& name) const;
    int   getInt(const QString& name) const;

    float offsetsHistogramSlope(int index) const;
    float offsetsHistogramThreshold(int index) const;

    void setFloat(const QString& name, float value);
    void setBool(const QString& name, bool value);
    void setInt(const QString& name, int value);

    bool  hasParam(QString param) const;

    static NLParameters& instance() { return _instance; }
    static NLParameters* instancePtr() { return &_instance; }

public slots:
    void styleChanged(int i);
    void setOffsetsHistogramSlope(double value);
    void setOffsetsHistogramThreshold(double value);

signals:
    void parametersChanged();
    void slopeChanged(double);
    void thresholdChanged(double);

protected:
    static NLParameters _instance;
    int _currentStyleIndex;

private:
    static QStringList float_names;
    static QStringList bool_names;
    static QStringList int_names;

    QHash<QString,float> _float_params;
    QHash<QString,bool>  _bool_params;
    QHash<QString,int>   _int_params;

    QVector<float> _offsetsHistogramSlopes;
    QVector<float> _offsetsHistogramThresholds;
};

#endif // NLPARAMETERS_H
