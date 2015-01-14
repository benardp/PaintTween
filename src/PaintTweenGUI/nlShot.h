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

#ifndef NLSHOTSETUP_H
#define NLSHOTSETUP_H

#include "nlInputElements.h"

#include <QtCore/QStringList>

class NLDataAccess;

class NLShot
{
    friend class NLDataAccess;

public:

    NLShot();

    void clear();

    // Function to setup a shot from an .xml file:
    bool setup(const QString& filename, const QDomElement &shotDataElt, 
               int firstFrame, int lastFrame);

    void save(QDomDocument& document, QDomElement &element) const;

protected:
    QString _unit;
    QString _shot;
    QString _shotDir;
    QString _workingDir;
    bool _fromFarm;

    int _width;
    int _height;

    NLInputElements _elements;
    QStringList _ribbonsF;
    QStringList _ribbonsB;
};

#endif
