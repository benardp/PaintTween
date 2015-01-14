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

#include "imagePyramid.h"
#include "cudaHostUtil.h"

#include <QtCore/QFileInfo>
#include <QtCore/QDataStream>

#include "imageIO.h"

static bool _preserve_cache = false;

void setPreserveImagePyramidCache(bool preserve)
{
    _preserve_cache = preserve;
}

bool getPreserveImagePyramidCache()
{
    return _preserve_cache;
}

QDataStream &operator<<(QDataStream &out, const Color4 &color)
{
    out << color.r;
    out << color.g;
    out << color.b;
    out << color.a;
    return out;
}

QDataStream &operator>>(QDataStream &in, Color4 &color)
{
    in >> color.r;
    in >> color.g;
    in >> color.b;
    in >> color.a;
    return in;
}

QDataStream &operator<<(QDataStream &out, const float2 &value)
{
    out << value.x;
    out << value.y;
    return out;
}

QDataStream &operator>>(QDataStream &in, float2 &value)
{
    in >> value.x;
    in >> value.y;
    return in;
}

template <class T>
void ImagePyramid<T>::initialize(int width, int height, const T& defaultValue, const QString& path, LayerDesc desc, DownsampleScaleMode scaleMode)
{
    _scaleMode = scaleMode;
    _baseWidth = width;
    _baseHeight = height;
    int numLevels = std::min(floor(log2f(_baseWidth)), floor(log2f(_baseHeight)));
    _storage.clear();
    for(int i=0; i<numLevels; i++){
	_storage.append(QVector<T>(this->width(i)*this->height(i),defaultValue));
    }
    _initialized = true;
    _cachePath = ImageIO::remapFilePath(path);
    _desc = desc;
}

template <class T>
void ImagePyramid<T>::initialize(const QVector<T> &base, int width, int height, const QString& path, LayerDesc desc, DownsampleScaleMode scaleMode)
{
    _scaleMode = scaleMode;
    _baseWidth = width;
    _baseHeight = height;
    downsamplePyramid(base, _storage, width, height, scaleMode);
    _initialized = true;
    _cachePath = ImageIO::remapFilePath(path);
    _desc = desc;
}

template <class T>
void ImagePyramid<T>::clear()
{
    if (!_preserve_cache) {
        cleanCache();
    }
    for(int i=0; i<numLevels(); i++){
        _storage[i].clear();
    }
    _storage.clear();
    _cachePath = QString();
    _baseWidth = _baseHeight = 0;
    _initialized = false;
    _desc = LayerDesc();
}

template <class T>
bool ImagePyramid<T>::load(int level)
{
    if(!isInitialized())
        return false;
    QString path = _cachePath.arg(level);
    QFileInfo info(path);

    int width, height;
    if(!info.exists() || !ImageIO::readImage(path,_desc,_storage[level],width,height)){
        return false;
    }
    return true;
}

template <class T>
bool ImagePyramid<T>::cacheOneLevel(int level, bool overwrite)
{
    bool success = true;
    if(!isLoaded(level))
        return success;
    QString path = _cachePath.arg(level);
    QFileInfo info(path);
    if(overwrite || !info.exists()){

        if(!ImageIO::writeImage(_storage[level],path,_desc,width(level),height(level))){
            qCritical("Can't open cache file for writing: %s",qPrintable(path));
            success = false;
        }
    }
    _storage[level].clear();
    return success;
}

template <class T>
bool ImagePyramid<T>::cache(int level, bool overwrite)
{
    if(!isInitialized())
        return false;

    bool success = true;
    if(level == -1) { //cache all levels
        for(int i=0; i<numLevels(); i++){
            success = success && cacheOneLevel(i,overwrite);
        }
    }else{
        success = cacheOneLevel(level,overwrite);
    }
    return success;
}

template <class T>
void ImagePyramid<T>::cleanCache()
{
    for(int i=0; i<numLevels(); i++){
        QFile file(_cachePath.arg(i));
        if(file.exists())
            file.remove();
    }
}

template <class T>
const T& ImagePyramid<T>::pixel(int x, int y, int level) const
{
    assert(!(x < 0 || x > width(level) || y < 0 || y > height(level)));

    return _storage.at(level).at(y * width(level) + x);
}

template <class T>
void ImagePyramid<T>::setPixel(const T& value, int x, int y, int level)
{
    _storage[level][y * width(level) + x] = value;
}

template <class T>
void ImagePyramid<T>::setLevel(const QVector<T>& data, int level)
{
    _storage[level] = QVector<T>(data);
}

template class ImagePyramid<int>;
template class ImagePyramid<float>;
template class ImagePyramid<float2>;
template class ImagePyramid<float3>;
template class ImagePyramid<Color4>;

void initializeRibbonPyramid(const QVector<RibbonP>& data, ImagePyramid<Color4>& pyramid, int width, int height, QString cachePath)
{
    QVector<Color4> downsampled;
    pyramid.initialize(width, height, Color4(), cachePath, LayerDesc("RibbonP","RibbonP",QStringList()<<"time_step"<<"y"<<"x"<<"layer"));
    downsampled.resize(width*height);
    for (int i = 0; i < width*height; i++) {
        downsampled[i] = data[i].toColor4();
    }
    pyramid.setLevel(downsampled, 0);
    for (int i = 1; i < pyramid.numLevels(); i++) {
        int w = pyramid.width(i);
        int h = pyramid.height(i);
        downsampled.resize(w*h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const RibbonP& p00 = RibbonP(pyramid.pixel(x*2,y*2,i-1));
                RibbonP out = p00;
                out.x *= 0.5;
                out.y *= 0.5;
                downsampled[y*w + x] = out.toColor4();
            }
        }
        pyramid.setLevel(downsampled, i);
    }
}
