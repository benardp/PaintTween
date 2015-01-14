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

#ifndef _IMAGE_PYRAMID_H_
#define _IMAGE_PYRAMID_H_

#include <QtCore/QFile>
#include <QtCore/QVector>
#include <QtCore/QStringList>

#include "downsample.h"

class LayerDesc
{
public:
    LayerDesc() : _layer_name(), _xml_name(), _channels() {}
    LayerDesc(const QString& layer_name, const QString& xml_name, const QStringList& channels)
        : _layer_name(layer_name), _xml_name(xml_name), _channels(channels) {}
    int numChannels() const { return _channels.size(); }
    QString _layer_name;
    QString _xml_name;
    QStringList _channels;
};

template <class T>
class ImagePyramid
{
public:
    ImagePyramid() : _storage(), _baseWidth(0), _baseHeight(0), _initialized(false) {}

    ~ImagePyramid() { clear(); }

    void initialize(int width, int height, const T &defaultValue, const QString &path = QString(),
		    LayerDesc desc = LayerDesc(), DownsampleScaleMode scaleMode = DOWNSAMPLE_SCALE_CONSTANT);
    void initialize(const QVector<T>& base, int width, int height, const QString &path = QString(),
                    LayerDesc desc = LayerDesc(), DownsampleScaleMode scaleMode = DOWNSAMPLE_SCALE_CONSTANT);

    bool isInitialized() const {
        return _initialized;
    }
    void clear();

    bool isLoaded(int level) const {
        return !_storage.at(level).isEmpty();
    }
    bool load(int level);
    bool cache(int level, bool overwrite = false);
    void cleanCache();

    int width(int level = 0) const { return _baseWidth >> level; }
    int height(int level = 0) const { return _baseHeight >> level; }
    int numLevels() const { return _storage.size(); }

    const T& pixel(int x, int y, int level = 0) const;
    void setPixel(const T& value, int x, int y, int level = 0);
    void setLevel(const QVector<T> &data, int level = 0);
    
    const QVector<T>& storage(int level) const { return _storage.at(level); }
    QVector<T>& storage(int level) { return _storage[level]; }
    int storageSize(int level) { return _storage.at(level).size(); }
    int storageSize(int level) const { return _storage.at(level).size(); }
    T* storagePointer(int level) { return _storage[level].data(); }
    const T* storagePointer(int level) const { return _storage.at(level).data(); }

    const T& at(int level, size_t i) const {return _storage.at(level).at(i);}
    QVector<T>& operator [] (int level) {return _storage[level];}

protected:
    bool cacheOneLevel(int level, bool write);

    QList< QVector<T> > _storage;
    int _baseWidth, _baseHeight;
    DownsampleScaleMode _scaleMode;
    bool _initialized;
    QString _cachePath;
    LayerDesc _desc;
};

// Nasty... could these be static in the class? -fcole apr 17 2012
void setPreserveImagePyramidCache(bool preserve);
bool getPreserveImagePyramidCache();

template<class T1, class T2>
bool pyramidSameSize(const ImagePyramid<T1>& a, const ImagePyramid<T2>& b) {
    return a.width() == b.width() && a.height() == b.height() && a.numLevels() == b.numLevels();
}

// Shared, type-specific helper functions for pyramids.
void initializeRibbonPyramid(const QVector<RibbonP>& data, ImagePyramid<Color4>& pyramid, int width, int height, QString cachePath);


#endif
