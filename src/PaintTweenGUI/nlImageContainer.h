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

#ifndef NLIMAGECONTAINER_H
#define NLIMAGECONTAINER_H

#include "PaintTweenCUDA/types.h"
#include "PaintTweenCUDA/dataAccess.h"

#include <QImage>
#include <QtCore/QString>
#include <QtCore/QVector>

//------------------------------------------------------------------------------
//
// PaintTween Image Container:	This is a wrapper for raw image data (raw vector of per-pixel data)
//
//------------------------------------------------------------------------------

class NLImageContainer
{

public:
    NLImageContainer();

    // Function to load an image:
    bool load(const QString& filePath, LayerDesc desc);

    // Function to create a zero image with specified size:
    void createZeroImage(int width, int height, int numChannels);

    // Function to clear the image:
    void clear();

    // Setter for the image data:
    void setImage(const QVector<float>& image, int width, int height);

    const QVector<float>& getImage() const;
    QVector<float>& getImage() {return _image; }

    // Getter for the file path:
    void setFilePath(const QString& path) {_filePath = path; }
    const QString& filePath() const { return _filePath; }

    // Getters for the width and height of the image:
    inline int width() const {return _width; }
    inline int height() const {return _height; }
    inline int initialWidth() const {return _initialWidth; }
    inline int initialHeight() const {return _initialHeight; }

    // Check if actual data is provided:
    inline bool empty() const {return _image.empty(); }

    // Check if image has same size as other image:
    bool hasSameSizeAs(const NLImageContainer& other) const;

    // Getters and setters for special purpose flags:
    inline bool isTransferred() const {return _transferred; }
    inline void setTransferred(bool transferred) {_transferred = transferred; }
    
    // Function to resize an image with white borders
    void padImage(int newWidth, int newHeight);
    
protected:
    // Image properties and raw data:

    // Original file path (for identification):
    QString _filePath;

    // Width and height of image:
    int _width;
    int _height;
    int _initialWidth;
    int _initialHeight;

    // Actual image representation:
    QVector<float> _image;
    LayerDesc _desc;

    QImage _qimage;

    // Sepcial purpose flags to mark the image:
    bool _transferred;
};

#endif
