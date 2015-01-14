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

#include "nlInputElements.h"

const LayerDesc _elements_desc[NUM_INPUT_ELEMENTS] =
{
    LayerDesc("rgba", "inputElem", QStringList() << "red" << "green" << "blue" << "alpha"),
    LayerDesc("orientation", "orientationElem", QStringList() << "Ix2" << "IxIy" << "Iy2"),
    LayerDesc("dist_trans", "distTransElem", QStringList() << "length"),
    LayerDesc("scale", "scaleElem", QStringList() << "x" << "y"),
    LayerDesc("id", "idMergedElem", QStringList() << "red" << "green" << "blue"),
    LayerDesc("surf_id", "surfIdElem", QStringList() << "u" << "v"),
    LayerDesc("vel_b_1", "velBElem", QStringList() << "x" << "y"),
    LayerDesc("vel_f_1", "velFElem", QStringList() << "x" << "y"),
    LayerDesc("vel_b_%1",  "ribbonBElem", QStringList() << "x" << "y"),
    LayerDesc("vel_f_%1",  "ribbonFElem", QStringList() << "x" << "y")
};

NLInputElements::NLInputElements()
    : _elements(NUM_INPUT_ELEMENTS), _present(NUM_INPUT_ELEMENTS,false), _path(NUM_INPUT_ELEMENTS)
{
}

void NLInputElements::clear()
{
    _elements.fill("");
    _present.fill(false);
    _path.fill("");
}

const QString& NLInputElements::elementName(InputElements elt)
{
    return _elements_desc[elt]._xml_name;
}

const LayerDesc& NLInputElements::layerDesc(InputElements elt)
{
    return _elements_desc[elt];
}

QString NLInputElements::get(InputElements elt) const
{
    return _elements.at(elt);
}

void NLInputElements::set(InputElements elt, const QString& name)
{
    _present[elt] = !name.isEmpty();
    _elements[elt] = name;
}

const QString& NLInputElements::getPath(InputElements elt) const
{
    return _path.at(elt);
}

void NLInputElements::setPath(InputElements elt, const QString& path)
{
    _path[elt] = path;
}

void NLInputElements::load(const QDomElement& element)
{
    // Function to read input elements (specification) from an .xml node:
    for(size_t i=0; i<NUM_INPUT_ELEMENTS; i++){
        if(element.hasAttribute(_elements_desc[i]._xml_name)){
            _present[i] = true;
            QString att = rmDots(element.attribute(_elements_desc[i]._xml_name));
            set((InputElements)i, att);
        }
    }
}

bool NLInputElements::save(QDomElement &element) const
{
    for(size_t i=0; i<NUM_INPUT_ELEMENTS; i++){
        if(_present.at(i))
            element.setAttribute(_elements_desc[i]._xml_name, _elements[i]);
    }

    return true;
}

QString NLInputElements::rmDots(const QString& elem) {
    // Helper function to remove leading and trailing dots:
    QString cleaned_elem = elem;

    if (cleaned_elem.isEmpty()){
        return cleaned_elem;
    }

    // Check first character, if it's a '.' remove it:
    if(cleaned_elem.startsWith('.'))
        cleaned_elem = cleaned_elem.remove(0,1);

    if (cleaned_elem.isEmpty()){
        return cleaned_elem;
    }

    // Also check last character, if it's a '.' remove it:
    if (cleaned_elem.endsWith('.')){
        cleaned_elem.chop(1);
    }

    return cleaned_elem;
}

QString NLInputElements::addDots(const QString &elem) {
    // Helper function to add leading and trailing dots:
    QString dotted_elem = elem;

    if (dotted_elem.isEmpty()){
        return dotted_elem;
    }

    // Check first character, if it's not a '.', add a '.':
    if(!dotted_elem.startsWith('.')){
        dotted_elem = dotted_elem.prepend('.');
    }

    // Then check if the last character is a '.', if not add a '.':
    if (!dotted_elem.endsWith('.')){
        dotted_elem = dotted_elem.append('.');
    }

    return dotted_elem;
}
