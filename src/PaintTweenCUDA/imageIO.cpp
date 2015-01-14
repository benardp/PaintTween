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

#include "imageIO.h"

#include <QtCore/QtDebug>
#include <QtCore/QDir>

#include <ImfHeader.h>
#include <ImfRgbaFile.h>
#include <ImfTiledRgbaFile.h>
#include <ImfInputFile.h>
#include <ImfTiledInputFile.h>
#include <ImfPreviewImage.h>
#include <ImfChannelList.h>
#include <ImfOutputFile.h>

using namespace Imath;
using namespace Imf;

const LayerDesc patchXFLayerDesc = LayerDesc("PatchXF","PatchXF",QStringList()<<"x"<<"y"<<"theta"<<"hyst"<<"layer"<<"lumShift_y"<<"lumShift_a"<<"scale_u"<<"scale_v");

void ImageIO::multithread(int numThreads)
{
    setGlobalThreadCount(numThreads);
}

QString ImageIO::remapFilePath(const QString& filePath)
{
    QHash<QString,QString> substitutions;
#ifdef _MSC_VER
    // Some hacks to make PaintTweenGUI work under windows.
    substitutions["/home/fcole/"] = "Z:/";
    substitutions["/scratch/tmp/"] = "C:/temp/";
#else
    substitutions["/scratch/tmp/"] = QDir::tempPath() + "/";
#endif

    QString working = filePath;
    QList<QString> keys = substitutions.keys();
    for (int i = 0; i < keys.size(); i++) {
        working = working.replace(keys[i], substitutions[keys[i]]);
    }

    return working;
}

QString ImageIO::netAddressablePath(const QString& hostname, const QString& local_path)
{
    // The pixar net does some remapping magic...
    QString outpath = local_path;
    if (local_path.startsWith("/usr/")) {
        QString prefix = "/host/data/" + hostname;
        QString trimmed = local_path.right(local_path.size()-4);
        outpath = prefix + trimmed;
    } else if (local_path.startsWith("/scratch/")) {
        QString prefix = "/net/" + hostname;
        outpath = prefix + local_path;
    }
    return outpath;
}

bool ImageIO::readImage(const QString& filePath,
                        Array<Rgba>& pixels,
                        int& width,
                        int &height)
{
    RgbaInputFile* in;
    try
    {
        QString remapped = remapFilePath(filePath);
        in = new RgbaInputFile(remapped.toStdString().c_str());
    }
    catch (const std::exception &e)
    {
        qWarning() << e.what();
        return false;
    }

    Header header = in->header();

    Box2i &dataWindow = header.dataWindow();
    int dw = dataWindow.max.x - dataWindow.min.x + 1;
    int dh = dataWindow.max.y - dataWindow.min.y + 1;
    int dx = dataWindow.min.x;
    int dy = dataWindow.min.y;

    pixels.resizeErase(dw * dh);
    in->setFrameBuffer(pixels - dx - dy * dw, 1, dw);

    try
    {
        in->readPixels (dataWindow.min.y, dataWindow.max.y);
    }
    catch (const std::exception &e)
    {
        delete in;
        qWarning() << e.what();
        return false;
    }

    width = dw;
    height = dh;

    delete in;
    return true;
}

bool ImageIO::readImage(const QString& filePath,
                        QVector<Color4>& colorImage,
                        int& width,
                        int &height)
{
    // Function to read an image from the given file path
    // and converting it into a Color4 image:

    Array<Rgba> pixels;
    bool success = readImage(filePath,pixels,width,height);
    if(!success)
        return false;

    colorImage.clear();
    colorImage.resize(width*height);
    Color4* data = colorImage.data();

    for(int i=0; i<width*height; i++){
        Rgba &p = pixels[i];
        data[i] = Color4((float)p.r,(float)p.g,(float)p.b,(float)p.a);
    }

    return true;
}


bool ImageIO::readImage(const QString& filePath,
                        QVector<PatchXF> &offsets,
                        int &width,
                        int &height)
{
    QVector<float> deepImage;
    if(!readImage(filePath,patchXFLayerDesc,deepImage,width,height))
	return false;

    offsets.clear();
    offsets.resize(width*height);
    PatchXF* data = offsets.data();
    int c = patchXFLayerDesc.numChannels();

    for(int i=0; i<width*height; ++i){
	data[i].x = deepImage.at(i*c);
	data[i].y = deepImage.at(i*c+1);
	data[i].theta = deepImage.at(i*c+2);
	data[i].hysteresis = deepImage.at(i*c+3);
	data[i].layer = deepImage.at(i*c+4);
	data[i].luminanceShift.y = deepImage.at(i*c+5);
	data[i].luminanceShift.a = deepImage.at(i*c+6);
	data[i].scaleU = deepImage.at(i*c+7);
	data[i].scaleV = deepImage.at(i*c+8);
    }

    return true;
}

bool ImageIO::readImage(const QString& filePath, QVector<RibbonP> &ribbon, int &width,int &height)
{
    // Function to read an image from the given file path,
    // using the specified color correction type using ice,
    // and converting it into a Color4 image:
    Array<Rgba> data1;
    bool success = readImage(filePath,data1,width,height);
    if(!success)
        return false;

    ribbon.clear();
    ribbon.resize(width*height);
    RibbonP* data = ribbon.data();

    for(int y=0; y<height; ++y){
        for(int x=0; x<width; ++x){
            Rgba &p = data1[x + y*width];
            Color4 c((float)p.r,(float)p.g,(float)p.b,(float)p.a);
            data[x + y*width] = RibbonP(c);
        }
    }

    return true;
}

bool ImageIO::readImage(const QString &filePath, const LayerDesc &desc, QVector<float> &pixels, int &width, int &height)
{
    if(desc._layer_name == "rgba"){
        Array<Rgba> image;
        bool success = readImage(filePath,image,width,height);
        if(!success)
            return false;

        pixels.clear();
        pixels.resize(width*height*4);
        float* data = pixels.data();
        for(int i=0; i<width*height; i++){
            data[4*i]   = (float)image[i].r;
            data[4*i+1] = (float)image[i].g;
            data[4*i+2] = (float)image[i].b;
            data[4*i+3] = (float)image[i].a;
        }
        return true;
    }

    try
    {
        InputFile file(qPrintable(remapFilePath(filePath)));

        Box2i dw = file.header().dataWindow();
        width = dw.max.x - dw.min.x + 1;
        height = dw.max.y - dw.min.y + 1;

        pixels.resize(width*height*desc.numChannels());

        FrameBuffer frameBuffer;

        for(int l= 0; l<desc.numChannels(); l++){
            frameBuffer.insert(qPrintable(desc._layer_name + "." + desc._channels.at(l)),
                               Slice(FLOAT, (char *) (pixels.data()) + l*sizeof(float),
                                     sizeof(float)*desc.numChannels(), sizeof(float)*width*desc.numChannels()));
        }

        file.setFrameBuffer(frameBuffer);
        file.readPixels (dw.min.y, dw.max.y);
    }
    catch (const std::exception &e)
    {
        qWarning() << e.what();
        return false;
    }

    return true;
}

bool ImageIO::readImage(const QString &filePath, const LayerDesc &desc, QVector<float2> &pixels, int &width, int &height)
{
    assert(desc.numChannels() == 2);

    QVector<float> image;
    bool success = readImage(filePath,desc,image,width,height);
    if(!success)
        return false;

    pixels.clear();
    pixels.resize(width*height);
    float2* data = pixels.data();
    for(int i=0; i<pixels.size(); i++)
        data[i] = make_float2(image.at(2*i),image.at(2*i+1));

    return true;
}

bool ImageIO::readImage(const QString &filePath, const LayerDesc &desc, QVector<float3> &pixels, int &width, int &height)
{
    assert(desc.numChannels() == 3);

    QVector<float> image;
    bool success = readImage(filePath,desc,image,width,height);
    if(!success)
        return false;

    pixels.clear();
    pixels.resize(width*height);
    float3* data = pixels.data();
    for(int i=0; i<pixels.size(); i++)
        data[i] = make_float3(image.at(3*i),image.at(3*i+1),image.at(3*i+2));

    return true;
}

bool ImageIO::readImage(const QString &filePath, const LayerDesc &desc, QVector<int> &pixels, int &width, int &height)
{
    assert(desc.numChannels() == 3);

    QVector<float> image;
    bool success = readImage(filePath,desc,image,width,height);
    if(!success)
        return false;

    pixels.clear();
    pixels.resize(width*height);
    int* data = pixels.data();
    for(int i=0; i<pixels.size(); i++){
        int r = floorf((image.at(3*i) * 255) + 0.5);
        int g = floorf((image.at(3*i+1) * 255) + 0.5);
        int b = floorf((image.at(3*i+2) * 255) + 0.5);

        data[i] = r + g*255 + b*255*255;
    }

    return true;
}

bool ImageIO::readImage(const QString &filePath, const LayerDesc &desc, QVector<Color4> &pixels, int &width, int &height)
{
    assert(desc.numChannels() == 4);

    QVector<float> image;
    bool success = readImage(filePath,desc,image,width,height);
    if(!success)
        return false;

    pixels.clear();
    pixels.resize(width*height);
    Color4* data = pixels.data();
    for(int i=0; i<pixels.size(); i++)
        data[i] = Color4(image.at(4*i),image.at(4*i+1),image.at(4*i+2),image.at(4*i+3));

    return true;
}

bool ImageIO::writeImage(const Array<Rgba>& pixels, const QString& filePath, int width, int height, Compression compression)
{
    try
    {
        RgbaOutputFile file(qPrintable(remapFilePath(filePath)), width, height, WRITE_RGBA, 1, Imath::V2f(0,0), 1, INCREASING_Y, compression);
        file.setFrameBuffer(pixels, 1, width);
        file.writePixels(height);
    }
    catch (const std::exception &e)
    {
        qWarning() << e.what();
        return false;
    }
    return true;
}

bool ImageIO::writeImage(const QVector<Color4>& image, const QString& filePath, int width, int height)
{
    // Function to write an image to the given file path,
    // relying on the input width and height arguments to
    // reflect the image dimensions:

    Array<Rgba> pixels;
    pixels.resizeErase(width * height);

    for(int i=0; i<width*height; i++){
        Color4 p = image.at(i);
        pixels[i] = Rgba((half)p.r,(half)p.g,(half)p.b,(half)p.a);
    }

    return writeImage(pixels,filePath,width,height);
}

bool ImageIO::writeImage(const QVector<PatchXF>& offsets, const QString& filePath, int width, int height)
{
    int c = patchXFLayerDesc.numChannels();
    QVector<float> deepImage(width*height*c);
    float* data = deepImage.data();
    for(int i=0; i<width*height; ++i){
	data[c*i]   = offsets.at(i).x;
	data[c*i+1] = offsets.at(i).y;
	data[c*i+2] = offsets.at(i).theta;
	data[c*i+3] = offsets.at(i).hysteresis;
	data[c*i+4] = offsets.at(i).layer;
	data[c*i+5] = offsets.at(i).luminanceShift.y;
	data[c*i+6] = offsets.at(i).luminanceShift.a;
	data[c*i+7] = offsets.at(i).scaleU;
	data[c*i+8] = offsets.at(i).scaleV;
    }
    return writeImage(deepImage,filePath,patchXFLayerDesc,width,height);
}

bool ImageIO::writeImage(const QVector<RibbonP>& ribbon, const QString& filePath, int width, int height)
{
    Array<Rgba> image1(ribbon.size());
    for(int y=0; y<height; ++y){
        for(int x=0; x<width; ++x){
            Color4 c = ribbon.at(x + y*width).toColor4();
            image1[x + y*width] = Rgba(c.r,c.g,c.b,c.a);
        }
    }
    return writeImage(image1,filePath,width,height);
}

bool ImageIO::writeImage(QVector<float> &pixels, const QString &filePath, const LayerDesc &desc, int width, int height)
{
    try
    {
        Imf::Header header (width, height);
        Imf::FrameBuffer frameBuffer;

        for (int chan = 0; chan < desc.numChannels(); chan++) {
            QString chan_name = QString("%1.%2").arg(desc._layer_name).arg(desc._channels[chan]);
            header.channels().insert(qPrintable(chan_name), Imf::Channel(Imf::FLOAT));
            frameBuffer.insert(qPrintable(chan_name), Imf::Slice(Imf::FLOAT, (char *) pixels.data() + chan*sizeof(float), sizeof(float)*desc.numChannels(), sizeof(float)*width*desc.numChannels()));
        }

        Imf::OutputFile file(qPrintable(remapFilePath(filePath)), header);
        file.setFrameBuffer(frameBuffer);

        file.writePixels(height);
    }
    catch (const std::exception &e)
    {
        qWarning() << e.what();
        return false;
    }
    return true;
}

bool ImageIO::writeImage(QVector<float2> &pixels, const QString &filePath, const LayerDesc &desc, int width, int height)
{
    assert(desc.numChannels() == 2);

    QVector<float> image;
    image.resize(pixels.size()*2);
    float* data = image.data();
    for(int i=0; i<pixels.size(); i++){
        float2 f = pixels.at(i);
        data[2*i]   = f.x;
        data[2*i+1] = f.y;
    }
    return writeImage(image,filePath,desc,width,height);
}

bool ImageIO::writeImage(QVector<float3> &pixels, const QString &filePath, const LayerDesc &desc, int width, int height)
{
    assert(desc.numChannels() == 3);

    QVector<float> image;
    image.resize(pixels.size()*3);
    float* data = image.data();
    for(int i=0; i<pixels.size(); i++){
        float3 f = pixels.at(i);
        data[3*i]   = f.x;
        data[3*i+1] = f.y;
        data[3*i+2] = f.z;
    }
   return  writeImage(image,filePath,desc,width,height);
}

bool ImageIO::writeImage(QVector<int> &pixels, const QString &filePath, const LayerDesc &desc, int width, int height)
{
    assert(desc.numChannels() == 3);

    QVector<float> image;
    image.resize(pixels.size()*3);
    float* data = image.data();
    for(int i=0; i<pixels.size(); i++){
        int value = pixels.at(i);
        double fvalue = (double)value / 255.0;
        data[3*i]   = fvalue - floor(fvalue);
        fvalue = (double)value / (255.0*255.0);
        data[3*i+1] = fvalue - floor(fvalue);
        fvalue = (double)value / (255.0*255.0*255.0);
        data[3*i+2] = fvalue - floor(fvalue);
    }
   return writeImage(image,filePath,desc,width,height);
}

bool ImageIO::writeImage(QVector<Color4> &pixels, const QString &filePath, const LayerDesc &desc, int width, int height)
{
    assert(desc.numChannels() == 4);

    if(desc._layer_name == "rgba"){
        return writeImage(pixels,filePath,width,height);
    }else{
        QVector<float> image;
        image.resize(pixels.size()*4);
        float* data = image.data();
        for(int i=0; i<pixels.size(); i++){
            Color4 c = pixels.at(i);
            data[4*i]   = c.r;
            data[4*i+1] = c.g;
            data[4*i+2] = c.b;
            data[4*i+3] = c.a;
        }
        return writeImage(image,filePath,desc,width,height);
    }
}
