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

#ifndef SYNTHESISIO_H
#define SYNTHESISIO_H

#if _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include "types.h"

#include <QString>

#include <ImfRgba.h>
#include <ImfArray.h>
#include <ImfCompression.h>

#include "dataAccess.h"

class ImageIO
{

public:

    static void multithread(int numThreads);

    static QString remapFilePath(const QString& filePath);

    static QString netAddressablePath(const QString& hostname, const QString& local_path);

    //----------------------------------------------------------
    // Image loading functions using OpenEXR

    static bool readImage(const QString& filePath, QVector<Color4>& colorImage,
                          int& width, int& height);

    static bool readImage(const QString& filePath, QVector<PatchXF>& offsets,
                          int& width, int& height);

    static bool readImage(const QString& filePath, QVector<RibbonP>& ribbon,
                          int& width, int& height);

    static bool readImage(const QString &filePath, const LayerDesc& desc,
                          QVector<float> &pixels, int &width, int &height);

    static bool readImage(const QString &filePath, const LayerDesc& desc,
                          QVector<float2> &pixels, int &width, int &height);

    static bool readImage(const QString &filePath, const LayerDesc& desc,
                          QVector<float3> &pixels, int &width, int &height);

    static bool readImage(const QString &filePath, const LayerDesc& desc,
                          QVector<int> &pixels, int &width, int &height);

    static bool readImage(const QString &filePath, const LayerDesc& desc,
                          QVector<Color4> &pixels, int &width, int &height);

    //----------------------------------------------------------
    // Image writing functions using OpenEXR

    static bool writeImage(const QVector<Color4>& image,
                           const QString& filePath, int width, int height);

    static bool writeImage(const QVector<PatchXF>& offsets,
                           const QString& filePath, int width, int height);

    static bool writeImage(const QVector<RibbonP>& ribbon,
                           const QString& filePath, int width, int height);

    static bool writeImage(QVector<float> &pixels, const QString &filePath,
                          const LayerDesc &desc, int width, int height);

    static bool writeImage(QVector<float2> &pixels, const QString &filePath,
                          const LayerDesc &desc, int width, int height);

    static bool writeImage(QVector<float3> &pixels, const QString &filePath,
                          const LayerDesc &desc, int width, int height);

    static bool writeImage(QVector<int> &pixels, const QString &filePath,
                          const LayerDesc &desc, int width, int height);

    static bool writeImage(QVector<Color4> &pixels, const QString &filePath,
                          const LayerDesc &desc, int width, int height);

    //----------------------------------------------------------

    static inline QString parsePath(const QString& filename, int index)
    {
        return filename.arg(index, 4, 10, QLatin1Char('0'));
    }

    static inline QString temporaryOutputPath(const QString& path, int level, int frame, int pass, int version)
    {
        QString suffix = QString("_%1_%2_%3").arg(level).arg(pass).arg(version);
        return path.arg(suffix).arg(frame, 4, 10, QLatin1Char('0'));
    }



private:

    static bool readImage(const QString& filePath, Imf::Array<Imf::Rgba> &pixels,
                          int& width, int& height);

    static bool writeImage(const Imf::Array<Imf::Rgba>& pixels,
                           const QString& filePath, int width, int height,
                           Imf::Compression compression = Imf::PIZ_COMPRESSION);
};

inline float strucTensorToAngle(const float& xx, const float& xy, const float& yy){
    float angle = 0.5*atan2f(2.0*xy, xx - yy) + M_PI_2;
    return angle >= M_PI_2 ? angle - M_PI : angle;
}

#endif
