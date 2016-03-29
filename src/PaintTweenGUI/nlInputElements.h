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

#ifndef NLINPUTELEMENTS_H
#define NLINPUTELEMENTS_H

#include <QString>
#include <QVector>
#include <QDomDocument>

#include "PaintTweenCUDA/dataAccess.h"

// !\class NLInputElements
// \brief Contains names of image elements used in the synthesis.

class NLInputElements
{

public:

    NLInputElements();

    void clear();

    void load(const QDomElement& element);
    bool save(QDomElement &element) const;

    QString get(InputElements elt) const;
    void set(InputElements elt, const QString& name);
    bool isPresent(InputElements elt) const { return _present.at(elt); }
    void setPresent(InputElements elt, bool b) { _present[elt] = b; }

    static const QString &elementName(InputElements elt);
    static const LayerDesc &layerDesc(InputElements elt);

    const QString &getPath(InputElements elt) const;
    void setPath(InputElements elt, const QString& path);

    // Helper function to remove leading and trailing dots:
    static QString rmDots(const QString& elem);
    static QString addDots(const QString& elem);

private:
    QVector<QString> _elements;
    QVector<bool>    _present;
    QVector<QString> _path;
};

#endif
