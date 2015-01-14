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

#include "nlStyles.h"

#include <QtCore/QDir>
#include <QtCore/QFile>

const LayerDesc _elements_desc[NUM_STYLE_ELEMENTS] =
{
    LayerDesc("style_output", "styleOutput", QStringList() << "red" << "green" << "blue" << "alpha"),
    LayerDesc("style_input", "styleInput", QStringList() << "red" << "green" << "blue" << "alpha"),
    LayerDesc("dist_trans", "styleDistanceTransform", QStringList() << "length"),
    LayerDesc("orientation", "styleOrientation", QStringList() << "Ix2" << "IxIy" << "Iy2"),
    LayerDesc("id", "styleObjectIDs", QStringList() << "red" << "green" << "blue")
};


NLStyles::NLStyles()
{
}

bool NLStyles::load(const QString& filename, const QDomElement &element)
{
    QDir fileroot = QFileInfo(filename).dir();

    QDomNode styleNode = element.firstChild();
    while(!styleNode.isNull()){
        if(styleNode.isElement()){
            QDomElement styleElt = styleNode.toElement();
            if(styleElt.hasAttribute("frame")){ //key-frame
                _keyFrameIndices.append(styleElt.attribute("frame").toInt());
            }
            QHash<StyleElements,NLImageContainer> newStyle;
            for(size_t i=0; i<NUM_STYLE_ELEMENTS; i++){
                QDomElement child = styleElt.firstChildElement(_elements_desc[i]._xml_name);
                if(!child.isNull() && child.hasAttribute("path")){
                    QString path = child.attribute("path");
                    path = QDir::cleanPath(fileroot.absoluteFilePath(path));
                    NLImageContainer image;
                    image.load(path,_elements_desc[i]);
                    newStyle.insert((StyleElements)i,image);
                }
            }
            if(!newStyle.isEmpty())
                _images.append(newStyle);
        }
        styleNode = styleNode.nextSibling();
        if(!_images.last().contains(STYLE_INPUT)){
            qCritical("No input style found!");
            return false;
        }
        if(!_images.last().contains(STYLE_OUTPUT)){
            qCritical("No output style found!");
            return false;
        }
    }
    return true;
}

void NLStyles::save(QDomDocument &document, QDomElement &element) const
{
    QDomElement styles = document.createElement("styles");
    element.appendChild(styles);

    for(int i=0; i<_images.size(); i++){
        QDomElement style = document.createElement("style");
        if(i<_keyFrameIndices.size()){
            style.setAttribute("frame",_keyFrameIndices.at(i));
        }
        QHashIterator<StyleElements,NLImageContainer> it(_images.at(i));
        while(it.hasNext()){
            it.next();
            QDomElement element = document.createElement(elementName(it.key()));
            element.setAttribute("path",it.value().filePath());
            style.appendChild(element);
        }
        styles.appendChild(style);
    }
}

const QString &NLStyles::elementName(StyleElements element) const
{
    return _elements_desc[element]._xml_name;
}

const LayerDesc &NLStyles::layerDesc(StyleElements element) const
{
    return _elements_desc[element];
}

bool NLStyles::haveSameSize() const
{
    bool success = true;
    for(int i=0; i<_images.size() && success; i++)
        for(int j=i+1; j<_images.size() && success; j++)
            success = (_images.at(i).size() == _images.at(j).size());

    for(int i=0; i<_images.size() && success; i++){
        QHashIterator<StyleElements,NLImageContainer> it(_images[i]);
        while (it.hasNext() && success) {
            it.next();
            success = (_images.at(i).value(STYLE_INPUT).width() == it.value().width());
            success = (_images.at(i).value(STYLE_INPUT).height() == it.value().height());
        }
    }
    if(!success)
        return false;
    return true;
}

void NLStyles::clear()
{
    _keyFrameIndices.clear();
    for(int i=0; i<_images.size(); ++i)
        _images[i].clear();
    _images.clear();
}

