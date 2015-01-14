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

#include "nlDataAccess.h"

#include <QtCore/QDir>

#include "PaintTweenCUDA/imageIO.h"

#if _MSC_VER
#define round(X) floor(X+0.5)
#endif

const QString _output_element_names [NUM_OUT_ELEMENTS] =
{
    "output", "outputFinal", "base", "offsets",
    "offsetsraw", "residual", "lumShift"
};

NLDataAccess::NLDataAccess()
{
    clear();
}

void NLDataAccess::clear()
{
    _firstFrame = 0;
    _lastFrame = 0;
    _loaded = false;
    _dataUptodate = false;
    _styles.clear();
    _shot.clear();

    // Make sure there is a temporary output directory.
    QString temp_out = ImageIO::remapFilePath("/scratch/tmp/texSynthTemp");
    if (!QDir::root().exists(temp_out)) {
        QDir::root().mkpath(temp_out);
    }
    _temporaryDir = QDir::home().absoluteFilePath(temp_out);
    _outputDir = _temporaryDir;
    _storeIntermediateImagesInTemp = true;
}

bool NLDataAccess::getInputElement(InputElements elt, int frame, NLImageContainer &image)
{
    if(!_shot._elements.isPresent(elt)){
        int width, height;
        getImageDimensions(width,height);
        _blank.createZeroImage(width,height,_shot._elements.layerDesc(elt).numChannels());
        image = _blank;
        return false;
    }

    QString path = _shot._elements.getPath(elt).arg(frame);
    return image.load(path,_shot._elements.layerDesc(elt));
}

void NLDataAccess::toOrientationPyramid(const ImagePyramid<float3> &fullPyramid, ImagePyramid<float> &pyramid)
{
    for (int l=0; l<fullPyramid.numLevels(); l++){
        float* data = pyramid[l].data();
        for (int i=0; i<fullPyramid.storageSize(l); i++){
            float Ix2 = fullPyramid.at(l,i).x;
            float IxIy = fullPyramid.at(l,i).y;
            float Iy2 = fullPyramid.at(l,i).z;
            data[i] = strucTensorToAngle(Ix2, IxIy, Iy2);
        }
    }
}

QString NLDataAccess::pyramidPath(InputElements elt, int frame) const
{
    return ImageIO::parsePath(getTemporaryDir()+"/image" + _shot._elements.elementName(elt) + "Pyramid_%2.%1.exr",frame);
}

QString NLDataAccess::pyramidPath(StyleElements elt) const
{
    return QString(getTemporaryDir()+"/exemplar" + _styles.elementName(elt) + "Pyramid_%1.exr");
}

template<class T>
bool NLDataAccess::getInputElement(InputElements elt, int frame, ImagePyramid<T> &pyramid, const T& defaultValue, DownsampleScaleMode mode)
{
    if(!_shot._elements.isPresent(elt)){
        pyramid.initialize(_shot._width, _shot._height, defaultValue, pyramidPath(elt,frame), _shot._elements.layerDesc(elt));
        return true;
    }

    int width, height;
    QVector<T> imageInput_img;
    QString path = _shot._elements.getPath(elt).arg(frame);
    if(!ImageIO::readImage(path, _shot._elements.layerDesc(elt), imageInput_img, width, height))
        return false;

    pyramid.initialize(imageInput_img, width, height, pyramidPath(elt,frame), _shot._elements.layerDesc(elt), mode);

    return true;
}

bool NLDataAccess::getInputElement(InputElements elt, int frame, ImagePyramid<Color4> &pyramid, DownsampleScaleMode mode)
{
    return getInputElement<Color4>(elt,frame,pyramid,mode);
}

bool NLDataAccess::getInputElement(InputElements elt, int frame, ImagePyramid<float2> &pyramid, DownsampleScaleMode mode)
{
    if(elt == IN_SCALE)
        return getInputElement<float2>(elt,frame,pyramid,make_float2(1.f,1.f),mode);

    return getInputElement<float2>(elt,frame,pyramid,make_float2(0.f,0.f),mode);
}

bool NLDataAccess::getInputElement(InputElements elt, int frame, ImagePyramid<float> &pyramid, DownsampleScaleMode mode)
{
    if(elt == IN_ORIENTATION){
        ImagePyramid<float3> fullPyramid;
        if(getInputElement<float3>(elt,frame,fullPyramid,make_float3(0.f,0.f,0.f),mode)){
            pyramid.initialize(fullPyramid.width(),fullPyramid.height(), 0.f, pyramidPath(elt,frame), LayerDesc("angle","angle",QStringList()<<"theta"));
            toOrientationPyramid(fullPyramid,pyramid);
            return true;
        }else{
            return false;
        }
    }
    return getInputElement<float>(elt,frame,pyramid,mode);
}

bool NLDataAccess::getInputElement(InputElements elt, int frame, ImagePyramid<int> &pyramid)
{
    return getInputElement<int>(elt,frame,pyramid,0);
}

bool NLDataAccess::getRibbonB(int frame, int step_size, ImagePyramid<float2> &pyramid)
{
    if (_shot._ribbonsB.empty())
        return false;

    int width, height;
    QVector<float2> image;
    int index = 0;
    for (; index < step_size; index++) {
        if (step_size == 1 << index) break;
    }
    if(index >= _shot._ribbonsB.size())
        return false;
    QString path = _shot._ribbonsB.at(index).arg(frame);
    LayerDesc desc = LayerDesc(_shot._elements.layerDesc(IN_RIBBON_B));
    desc._layer_name = desc._layer_name.arg(step_size);
    if (!ImageIO::readImage(path, desc, image, width, height))
        return false;

    QString cacheName = QString("/imageRibbonB%1Pyramid_%3.%2.exr").arg(step_size);

    pyramid.initialize(image, width, height, ImageIO::parsePath(getTemporaryDir()+cacheName,frame), desc, DOWNSAMPLE_SCALE_WITH_LEVELS);

    return true;
}

bool NLDataAccess::getRibbonF(int frame, int step_size, ImagePyramid<float2> &pyramid)
{
    if (_shot._ribbonsF.empty())
        return false;

    int width, height;
    QVector<float2> image;
    int index = 0;
    for (; index < step_size; index++) {
        if (step_size == 1 << index) break;
    }
    if(index >= _shot._ribbonsF.size())
        return false;
    QString path = _shot._ribbonsF.at(index).arg(frame);
    LayerDesc desc = LayerDesc(_shot._elements.layerDesc(IN_RIBBON_F));
    desc._layer_name = desc._layer_name.arg(step_size);
    if (!ImageIO::readImage(path, desc, image, width, height))
        return false;

    QString cacheName = QString("/imageRibbonF%1Pyramid_%3.%2.exr").arg(step_size);

    pyramid.initialize(image, width, height, ImageIO::parsePath(getTemporaryDir()+cacheName,frame), desc, DOWNSAMPLE_SCALE_WITH_LEVELS);
    return true;
}

int NLDataAccess::getMaxRibbonStep()
{
    return 1 << (_shot._ribbonsF.size() - 1);
}

void NLDataAccess::getImageDimensions(int &width, int &height)
{
    if((_shot._width == -1 || _shot._height == -1) && _loaded){
        QString path = _shot._elements.getPath(IN_INPUT).arg(firstFrame());
        QVector<Color4> imageInput_img;
        ImageIO::readImage(path, imageInput_img, _shot._width, _shot._height);
    }

    width = _shot._width;
    height = _shot._height;
}

int NLDataAccess::firstFrame() const
{
    return _firstFrame;
}

int NLDataAccess::lastFrame() const
{
    return _lastFrame;
}

void NLDataAccess::setFirstFrame(int frame)
{
    _firstFrame = frame;
    _curPreviewFrame = _firstFrame;
    _dataUptodate = false;
}

void NLDataAccess::setLastFrame(int frame)
{
    _lastFrame = frame;
    _dataUptodate = false;
}

SynthesisScheme NLDataAccess::getSynthesisScheme() const
{
    return (SynthesisScheme)NLParameters::instance().getInt("synthesisScheme");
}

PassDirection NLDataAccess::getFirstPassDirection() const
{
    return (PassDirection)NLParameters::instance().getInt("firstPassDirection");
}

float NLDataAccess::getFloatParameter(const QString &param) const
{
    return NLParameters::instance().getFloat(param);
}

int NLDataAccess::getIntParameter(const QString &param) const
{
    return NLParameters::instance().getInt(param);
}

bool NLDataAccess::getBoolParameter(const QString &param) const
{
    return NLParameters::instance().getBool(param);
}

const QVector<float> &NLDataAccess::getOffsetsHistogramSlopes() const
{
    return NLParameters::instance()._offsetsHistogramSlopes;
}

const QVector<float> &NLDataAccess::getOffsetsHistogramThresholds() const
{
    return NLParameters::instance()._offsetsHistogramThresholds;
}

TsParameters NLDataAccess::getTsDefaultParams() const
{
    return NLParameters::instance().getTsDefaultParams();
}

int NLDataAccess::getNumStyles() const
{
    return _styles._images.size();
}

int NLDataAccess::getStyleWidth(int i) const
{
    assert(i < getNumStyles() && _styles._images.at(i).contains(STYLE_OUTPUT));
    return _styles._images.at(i)[STYLE_OUTPUT].width();
}

int NLDataAccess::getStyleHeight(int i) const
{
    assert(i < getNumStyles() && _styles._images.at(i).contains(STYLE_OUTPUT));
    return _styles._images.at(i)[STYLE_OUTPUT].height();
}

void NLDataAccess::getMaxStyleDimensions(int &maxWidth, int &maxHeight) const
{
    maxWidth = 0; maxHeight = 0;
    for(int i = 0; i < getNumStyles(); ++i){
        if(getStyleWidth(i) > maxWidth)
            maxWidth = getStyleWidth(i);

        if(getStyleHeight(i) > maxHeight)
            maxHeight = getStyleHeight(i);
    }
}

void NLDataAccess::getStyle(StyleElements element, int i, QVector<Color4> &style_img)
{
    assert(_styles.layerDesc(element).numChannels() >= 4);

    NLImageContainer image;
    getStyle(element,i,image);
    QVector<float> img = image.getImage();

    style_img.resize(img.size()/4);
    Color4* data = style_img.data();
    for(int i=0; i<style_img.size(); i++)
        data[i] = Color4(img.at(4*i),img.at(4*i+1),img.at(4*i+2),img.at(4*i+3));
}

void NLDataAccess::getStyle(StyleElements element, int i, NLImageContainer &image)
{
    assert(i < getNumStyles());

    if(_styles._images.at(i).contains(element))
        image = _styles._images.at(i)[element];
    else{
        _blank.createZeroImage(getStyleWidth(i),getStyleHeight(i),_styles.layerDesc(element).numChannels());
        image = _blank;
    }
}

void NLDataAccess::getStyle(StyleElements element, int i, ImagePyramid<Color4> &pyramid)
{
    NLImageContainer image;
    getStyle(element,i,image);
    QVector<float> img = image.getImage();
    QVector<Color4> style_img;
    style_img.resize(img.size()/4);
    Color4* data = style_img.data();
    for(int i=0; i<style_img.size(); i++)
        data[i] = Color4(img.at(4*i),img.at(4*i+1),img.at(4*i+2),img.at(4*i+3));

    pyramid.initialize(style_img, image.width(), image.height(), pyramidPath(element), _styles.layerDesc(element));
}

void NLDataAccess::getStyle(StyleElements element, int i, ImagePyramid<float> &pyramid)
{
    NLImageContainer image;
    getStyle(element,i,image);
    QVector<float> img = image.getImage();

    if(element == STYLE_ORIENTATION){
        QVector<float3> style_img;
        style_img.resize(img.size()/3);
        float3* data = style_img.data();
        for(int i=0; i<style_img.size(); i++)
            data[i] = make_float3(img.at(3*i),img.at(3*i+1),img.at(3*i+2));
        ImagePyramid<float3> fullPyramid;
        fullPyramid.initialize(style_img,image.width(),image.height(), pyramidPath(element));
        pyramid.initialize(image.width(), image.height(), 0.f, pyramidPath(element), LayerDesc("angle","angle",QStringList()<<"theta"));
        toOrientationPyramid(fullPyramid,pyramid);
    }else{
        pyramid.initialize(img,image.width(),image.height(), pyramidPath(element), _styles.layerDesc(element));
    }
}

void NLDataAccess::getStyle(StyleElements element, int i, ImagePyramid<int> &pyramid)
{
    NLImageContainer image;
    getStyle(element,i,image);
    QVector<float> style_img = image.getImage();

    if(element == STYLE_OBJECT_ID){
        QVector<int> imageId_input;
        imageId_input.clear();
        imageId_input.resize(image.width()*image.height());
        int* data = imageId_input.data();
        for(int i=0; i<imageId_input.size(); i++){
            int r = round(style_img.at(3*i) * 255);
            int g = round(style_img.at(3*i+1) * 255);
            int b = round(style_img.at(3*i+2) * 255);

            data[i] = r + g*255 + b*255*255;
        }
        pyramid.initialize(imageId_input, image.width(), image.height(),pyramidPath(element), _styles.layerDesc(element));
    }
}

void NLDataAccess::padStyles(int newWidth, int newHeight)
{
    for(int i=0; i<getNumStyles(); i++){
        for(int j=0; j<NUM_STYLE_ELEMENTS; j++){
            if(_styles._images.at(i).contains((StyleElements)j))
                _styles._images[i][(StyleElements)j].padImage(newWidth,newHeight);
        }
    }
}

const QVector<int> & NLDataAccess::keyFrameIndices() const
{
    return _styles._keyFrameIndices;
}

QString NLDataAccess::getOutDir() const
{
    return _outputDir;
}

QString NLDataAccess::getTemporaryDir() const
{
    if (_storeIntermediateImagesInTemp)
        return _temporaryDir;
    return _outputDir;
}

QString NLDataAccess::getOutPath(OutputElements element) const
{
    QString path = getTemporaryDir();
    if(element == OUT_OUTPUT_FINAL)
        path = _outputDir;
    return  path + "/" + _output_element_names[element] + "%1.%2.exr";
}

int NLDataAccess::version() const
{
    if(NLParameters::instance()._int_params.contains("version"))
        return NLParameters::instance()._int_params["version"];
    return 0;
}


bool NLDataAccess::isStyleReady(QString& msg) const
{
    // Check if style data is correcly sized:
    if (!_styles.haveSameSize()){
        msg = QString("Style images must have the same dimensions (width/height)!");
        return false;
    }

    // Check if any style data has been provided:
    if(_styles._images.isEmpty()) { // after above call, we know all these vectors are same size, so just check one
        msg = QString("Style images not provided!!");
        return false;
    }

    return true;
}

bool NLDataAccess::isAnimReady(QString &msg) const
{
    // Check if animation input paths have been provided:
    if(!_shot._elements.isPresent(IN_INPUT)){
        msg = QString("Animation input not loaded yet!");
        return false;
    }

    for(int i=1; i<NUM_INPUT_ELEMENTS; i++){
        if (!_shot._elements.isPresent((InputElements)i)){
            msg = QString("Animation %1 not loaded yet!").arg(_shot._elements.elementName((InputElements)i));
        }
    }

    return true;
}

bool NLDataAccess::goToFrame(int frame)
{
    if(frame == _curPreviewFrame)
        return true;

    QString msg;
    if (isAnimReady(msg)){
        if (frame >= _firstFrame && frame <= _lastFrame){
            _curPreviewFrame = frame;
            _dataUptodate = false;
            return true;
        }
    }
    qCritical(qPrintable(msg));
    return false;
}

QString NLDataAccess::getWorkingDir() const
{
    return _shot._workingDir;
}
