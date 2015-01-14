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

#ifndef NLSTYLES_H
#define NLSTYLES_H

#include <QtCore/QVector>
#include <QtCore/QHash>
#include <QtXml/QDomDocument>

#include "PaintTweenCUDA/dataAccess.h"
#include "nlImageContainer.h"

class NLDataAccess;

class NLStyles
{
    friend class NLDataAccess;

public:
    NLStyles();

    bool load(const QString& filename, const QDomElement &element);
    void save(QDomDocument& document, QDomElement &element) const;

    const QString& elementName(StyleElements element) const;
    const LayerDesc& layerDesc(StyleElements element) const;

    bool haveSameSize() const;

    void clear();

private:
    QList< QHash<StyleElements,NLImageContainer> > _images;
    QVector<int> _keyFrameIndices;
};

#endif // NLSTYLES_H
