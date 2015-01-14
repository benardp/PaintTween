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

#include "nlImageContainer.h"

#include "PaintTweenCUDA/imageIO.h"

#include "QtGui/QColor"
#include "QtCore/QString"

#include <iostream>


inline float unit_clamp(float input){
    if (input > 1.0f){
        return 1.0f;
    }
    if (input < 0.0f){
        return 0.0f;
    }
    return input;
}

NLImageContainer::NLImageContainer()
    : _width(0),
      _height(0),
      _initialWidth(0),
      _initialHeight(0),
      _transferred(false)
{
}

// Function to load an image:
bool NLImageContainer::load(const QString &filePath, LayerDesc desc)
{
    _width = 0;
    _height = 0;
    _image.clear();
    _filePath.clear();
    _desc = desc;

    bool success = ImageIO::readImage(filePath, desc, _image, _width, _height);
    if (!success || _image.empty()){
        return false;
    }

    _initialWidth = _width;
    _initialHeight = _height;

    // Remember the file path:
    _filePath = filePath;
    _transferred = false;
    return true;
}

// Function to create a zero image with specified size:
void NLImageContainer::createZeroImage(int width, int height, int numChannels)
{
    _image.resize(width*height*numChannels);
    _filePath.clear();
    _width = width;
    _height = height;
    _initialWidth = width;
    _initialHeight = height;
    _transferred = false;
    QStringList channels;
    for(int i=0; i<numChannels; i++) channels << QString("z%1").arg(i);
    _desc = LayerDesc("zero","zero",channels);
}

// Function to clear the image:
void NLImageContainer::clear()
{
    _image.clear();
    _filePath.clear();
    _width = 0;
    _height = 0;
    _initialWidth = 0;
    _initialHeight = 0;
    _transferred = false;
}

// Check if image has same size as other image:
bool NLImageContainer::hasSameSizeAs(const NLImageContainer& other) const
{
    if (_width != other.width()){
        return false;
    }
    if (_height != other.height()){
        return false;
    }
    return true;
}

// Function to resize an image with white borders 
void NLImageContainer::padImage(int newWidth, int newHeight)
{
    // don't do anything if we're resizing to the same size
    if ( (newWidth == _width) &&  (newHeight == _height))
        return;

    // warn user if they're cropping an image (probably they don't want to do this.)
    if ( (newWidth < _width) || (newHeight < _height)){
        std::cerr<<"NLImage Container: WARNING - image resize requested with new dimension smaller than original. Image will be cropped.\n";
    }

    // black, transparent pixels on the border.
    QVector<float> newImage(newWidth * newHeight * _desc.numChannels(),0.f);
    float* data = newImage.data();
    // copy the old values into the new image, as bounds permit.
    for(int y = 0; y < std::min(newHeight, _height); ++y){
        for(int x = 0; x < std::min(newWidth, _width); ++x){
            for(int l = 0; l < _desc.numChannels(); ++l){
                data[x + l + y * newWidth * _desc.numChannels()] = _image.at(x + l + y * _width * _desc.numChannels());
            }
        }
    }

    // now make this new image the _image.
    _image = newImage;
    _initialWidth = _width;
    _width = newWidth;
    _initialHeight = _height;
    _height = newHeight;
}
