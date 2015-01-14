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

#ifndef NLDIALSANDKNOBS_H
#define NLDIALSANDKNOBS_H

#include <QtGui/QWidget>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QSpinBox>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QLabel>
#include <QtGui/QHBoxLayout>
#include <QtCore/QVariant>

#include "nlParameters.h"

class NLDialsAndKnobs;

class NLVariant : public QObject
{
    Q_OBJECT
public:
    NLVariant(QLayout *parent, const QString& paramName, const QString& text);

public slots:
    virtual void update() = 0;

protected:
    QString _paramName;
    QLabel* _label;
    QHBoxLayout* _layout;

    friend class NLDialsAndKnobs;
};

class NLFloat : public NLVariant
{
    Q_OBJECT

public:
    NLFloat(QLayout* parent, const QString& paramName, const QString& text);
    NLFloat(QLayout* parent, const QString& paramName, const QString& text,
            double lower_limit, double upper_limit, double step_size);

    float value() const;

public slots:
    void setValue(double f);
    virtual void update();

signals:
    void valueChanged(float f);

protected:
    QDoubleSpinBox* _spinBox;

    friend class NLDialsAndKnobs;
};

class NLBool : public NLVariant
{
    Q_OBJECT

public:
    NLBool(QLayout* parent, const QString& paramName, const QString& text);

    bool value() const;

public slots:
    void setValue(int s);
    virtual void update();

signals:
    void valueChanged(bool b);

protected:
    QCheckBox* _checkBox;

    friend class NLDialsAndKnobs;
};

class NLInt : public NLVariant
{
    Q_OBJECT

public:
    NLInt(QLayout* parent, const QString& paramName, const QString& text);
    NLInt(QLayout* parent, const QString& paramName, const QString& text,
          int lower_limit, int upper_limit, int step_size);

    int value() const;

public slots:
    void setValue(int i);
    virtual void update();

signals:
    void valueChanged(int i);

protected:
    QSpinBox* _spinBox;

    friend class NLDialsAndKnobs;
};

class NLStringList : public NLVariant
{
    Q_OBJECT

public:
    NLStringList(QLayout* parent, const QString& paramName, const QString& text, QStringList& choices);

    int value() const;

public slots:
    void setValue(int i);
    virtual void update();

signals:
    void valueChanged(int i);

protected:
    QComboBox* _comboBox;

    friend class NLDialsAndKnobs;
};

class NLDialsAndKnobs : public QWidget
{
    Q_OBJECT

public:
    NLDialsAndKnobs(QWidget *parent = 0, Qt::WindowFlags flags = 0);
    ~NLDialsAndKnobs();

private:

    QList<NLVariant*> _knobs;

    // historgram Parameters
    QDoubleSpinBox* _offsetsHistogramSlope;
    QDoubleSpinBox* _offsetsHistogramThreshold;
};

#endif // NLDIALSANDKNOBS_H
