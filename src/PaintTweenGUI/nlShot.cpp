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

#include "nlShot.h"

#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QFile>

#include "PaintTweenCUDA/dataAccess.h"
#include "PaintTweenCUDA/imageIO.h"

NLShot::NLShot()
    : _width(-1), _height(-1)
{
}

void NLShot::clear()
{
    _unit.clear();
    _shot.clear();
    _shotDir.clear();
    _workingDir.clear();
    _width = -1;
    _height = -1;
    _elements.clear();
    _ribbonsF.clear();
    _ribbonsB.clear();
}

bool NLShot::setup(const QString& filename, const QDomElement& shotDataElt, 
                   int firstFrame, int lastFrame)
{
    QDir fileroot = QFileInfo(filename).dir();
    // Read the .xml element and call the setup function:

    // Read unit:
    QDomElement unitElt = shotDataElt.firstChildElement("unit");
    if (unitElt.isNull()){
        qWarning() << "Unit tag not found in xml file.";
        return false;
    }
    if(unitElt.hasAttribute("name")){
        _unit = unitElt.attribute("name");
    }else{
        qWarning() << "Unit name not found in xml file.";
        return false;
    }

    // Read shot:
    QDomElement shotElt = shotDataElt.firstChildElement("shot");
    if (shotElt.isNull()){
        qWarning() << "Shot tag not found in xml file.";
        return false;
    }
    if(shotElt.hasAttribute("name")){
        _shot = shotElt.attribute("name");
    }else{
        qWarning() << "Shot name not found in xml file.";
        return false;
    }

    // Read shot directory.
    QDomElement shotDirElt = shotDataElt.firstChildElement("shotDir");
    if(shotDirElt.isNull()){
        qWarning() << "Shot dircteory not found in xml file.";
        return false;
    }
    if(shotDirElt.hasAttribute("path")){
        _shotDir = shotDirElt.attribute("path");
        _shotDir = QDir::cleanPath(fileroot.absoluteFilePath(_shotDir));
    }else{
        qWarning() << "Shot path not found in xml file.";
        return false;
    }

    // Read working directory:
    QDomElement workingDirElt = shotDataElt.firstChildElement("workingDir");
    if (workingDirElt.isNull()){
        qWarning() << "workingDir path not found in xml file.";
        return false;
    }
    if(workingDirElt.hasAttribute("path")){
        _workingDir = workingDirElt.attribute("path");
        _workingDir = QDir::cleanPath(fileroot.absoluteFilePath(_workingDir));
    }else{
        qWarning() << "workingDir path not found in xml file.";
        return false;
    }

    QDomElement elementsElt = shotDataElt.firstChildElement("elements");
    if(elementsElt.isNull()){
        qWarning() << "elements tag not found in xml file.";
        return false;
    }
    _elements.load(elementsElt);

    // Read flags:
    QDomElement fileAccessElt = shotDataElt.firstChildElement("fileAccess");
    if(fileAccessElt.isNull()){
        qWarning() << "access tag not found in xml file.";
        return false;
    }
    if(fileAccessElt.hasAttribute("checkFarm")){
        _fromFarm = fileAccessElt.attribute("checkFarm").toInt() == 1;
    }

    QString root;
    if(_fromFarm){
        root = QDir::cleanPath(_shotDir);
    }else{
        root = QDir::cleanPath(_shotDir + "/" + _shot);
    }

    for(int i=0; i < IN_RIBBON_B; i++){
        if(_elements.isPresent((InputElements)i)){
            QString path = root;
            if(_fromFarm)
                path = path + "/" + _elements.get((InputElements)i) + "/" + _shot;
            path += NLInputElements::addDots(_elements.get((InputElements)i)) + "%1.exr";
            path = ImageIO::remapFilePath(path);

            for(int f = firstFrame; f <= lastFrame; f++){
                QFile filePtr(path.arg(f));

                if (!filePtr.exists()){
                    qWarning() << "Shot Setup: cannot load input " << path.arg(f);
                    if(i == IN_INPUT)
                        return false;
                    _elements.setPresent((InputElements)i,false);
                    break;
                }
            }
            _elements.setPath((InputElements)i, path);
        }
    }

    for (int i = 0; i < 10; i++) {
        int step_size = 1 << i;
        QString elemB = _elements.get(IN_RIBBON_B).replace(QRegExp("\\[.*\\]"), QString("%1").arg(step_size, 2, 10, QChar('0')));
        QString elemF = _elements.get(IN_RIBBON_F).replace(QRegExp("\\[.*\\]"), QString("%1").arg(step_size, 2, 10, QChar('0')));
        if(elemB.isEmpty() || elemF.isEmpty())
            continue;
        QString path = root;
        if(_fromFarm){
            elemB = path + "/" + elemB + "/" + _shot + "." + elemB;
            elemF = path + "/" + elemF + "/" + _shot + "." + elemF;
        }
        elemB += ".%1.exr";
        elemF += ".%1.exr";
        bool not_found = false;
        for(int f = firstFrame; f <= lastFrame; f++){
            QFile filePtrF(elemB.arg(f));
            QFile filePtrB(elemF.arg(f));

            if (!filePtrF.exists() || !filePtrB.exists()){
                //qWarning() << "Shot Setup: cannot load ribbon "<< i;
                not_found = true;
                break;
	    }
        }
        if (not_found)
            break;
        _ribbonsF.append(elemF);
        _ribbonsB.append(elemB);
    }

    qWarning() << "Shot Setup: loaded " << _ribbonsF.size() << " ribbon steps";

    return true;
}

void NLShot::save(QDomDocument &document, QDomElement &element) const
{
    QDomElement anim = document.createElement("anim");
    element.appendChild(anim);

    QDomElement unit = document.createElement("unit");
    unit.setAttribute("name",_unit);
    anim.appendChild(unit);

    QDomElement shot = document.createElement("shot");
    shot.setAttribute("name",_shot);
    anim.appendChild(shot);

    QDomElement shotDir = document.createElement("shotDir");
    shotDir.setAttribute("path",_shotDir);
    anim.appendChild(shotDir);

    QDomElement workingDir = document.createElement("workingDir");
    workingDir.setAttribute("path",_workingDir);
    anim.appendChild(workingDir);

    QDomElement elements = document.createElement("elements");
    _elements.save(elements);
    anim.appendChild(elements);

    QDomElement fileAccess = document.createElement("fileAccess");
    if(_fromFarm){
        fileAccess.setAttribute("checkFarm",1);
        fileAccess.setAttribute("checkFiles",0);
    }else{
        fileAccess.setAttribute("checkFarm",0);
        fileAccess.setAttribute("checkFiles",1);
    }
    anim.appendChild(fileAccess);
}
