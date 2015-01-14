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

#include <cuda.h>
#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>
#include "types.h"
#include "cudaHostUtil.h"

#include "cudaImagePyramid.h"

#include <QtCore/QDebug>

#include <cstdio>
#include <cstring>

#include "texSynth_kernel.h"



std::vector<CudaImagePyramidHost*> CudaImagePyramidHost::_instances;

CudaImagePyramidHost::CudaImagePyramidHost(int textureType, int typeSize, const char *name) :
    _storage(NULL), _typeSize(typeSize), _textureType(textureType),
    _baseWidth(0), _baseHeight(0), _numLayers(1), _in_destructor(false)
{
    strncpy(_name, name, 200);
    // By convention, the associated Cuda texture is (name)_TEXTURE_PYRAMID
    snprintf(_texture_name, 200, "%s_TEXTURE_PYRAMID", _name);
}

CudaImagePyramidHost::~CudaImagePyramidHost()
{
    _in_destructor = true;
    clear();
}

void CudaImagePyramidHost::initialize(int width, int height, cudaTextureFilterMode filter_mode, int depth)
{
    if (isInitialized() && width == _baseWidth && height == _baseHeight && filter_mode == _filterMode) {
        return;
    }

    clear();

    _baseWidth = width;
    _baseHeight = height;
    _filterMode = filter_mode;
    _numLayers = depth;

    // Get the texture and its channel descriptor to allocate the storage.
    const textureReference* constTexRefPtr=NULL;
	const void* constTexRef = TexSynth::getTextureReferenceByName(_texture_name);
    cudaGetTextureReference(&constTexRefPtr, constTexRef);
    checkCUDAError("Can't get tex ref for init", _texture_name);
    cudaChannelFormatDesc formatDesc = constTexRefPtr->channelDesc;

    if(_textureType == cudaTextureType2DLayered){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,0);
        if(prop.maxTexture2DLayered[0] < _baseWidth || prop.maxTexture2DLayered[1] < _baseHeight || prop.maxTexture2DLayered[2] < _numLayers){
            qDebug()<< "Max layered texture size:" << prop.maxTexture2DLayered[0] << " x " << prop.maxTexture2DLayered[1] << " x " << prop.maxTexture2DLayered[2];
            assert(0);
        }
        cudaExtent extent = make_cudaExtent(_baseWidth, _baseHeight, _numLayers);
        cudaMalloc3DArray(&_storage, &formatDesc, extent, cudaArrayLayered);
    }else{
        cudaMallocArray(&_storage, &formatDesc, _baseWidth, _baseHeight);
    }
    checkCUDAError("Failure to allocate", _name);

    // Set texture parameters.
    // Evil hack to get around an apparent bug in the cuda api:
    // cudaGetTextureReference only returns a const reference, and
    // there is no way to set the parameters with a reference other
    // than cast it to non-const.
    textureReference* texRefPtr=NULL;
    texRefPtr = const_cast<textureReference*>( constTexRefPtr );
    texRefPtr->addressMode[0] = cudaAddressModeClamp;
    texRefPtr->addressMode[1] = cudaAddressModeClamp;
    texRefPtr->filterMode = filter_mode;
    texRefPtr->normalized = false; // Use unnormalized (pixel) coordinates for addressing. This forbids texture mode wrap.

    bindTexture();

    bool found = false;
    for (size_t i = 0; i < _instances.size(); i++) {
        if (_instances[i] == this)
            found = true;
    }
    if (!found) {
        _instances.push_back(this);
    }
}

void CudaImagePyramidHost::clear() 
{
    if (!isInitialized()) {
        return;
    }

    // Don't bother unbinding the texture if everything is getting destroyed,
    // because it's likely that CUDA has already destroyed the texture.
    if (!_in_destructor) {
        unbindTexture();
    }

    cudaFreeArray(_storage);
    checkCUDAError("Free error", _name);

    _storage = NULL;
    _baseWidth = 0; _baseHeight = 0; _baseWidth = 0; _baseHeight = 0;
}

void CudaImagePyramidHost::clearAllInstances()
{
    for (size_t i = 0; i < _instances.size(); i++) {
        _instances[i]->clear();
    }
}

void CudaImagePyramidHost::copyFromHost(const void* source)
{
    assert(isInitialized());
    assert(_textureType == cudaTextureType2D);

    cudaMemcpyToArray(_storage, 0,0, source, _baseWidth*_baseHeight*_typeSize, cudaMemcpyHostToDevice);

    checkCUDAError("Memcpy error", _name);
}

void CudaImagePyramidHost::copyFromHost(int width, int height, const void* source, int layer)
{
    assert(isInitialized());
    assert(_textureType == cudaTextureType2DLayered);

    cudaMemcpy3DParms myParms = {0};
    myParms.srcPtr = make_cudaPitchedPtr((void*)source,width*_typeSize,width,height);
    myParms.srcPos = make_cudaPos(0,0,0);
    myParms.dstArray = _storage;
    myParms.dstPos = make_cudaPos(0,0,layer);
    myParms.extent = make_cudaExtent(width,height,1);
    myParms.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&myParms);

    checkCUDAError("Memcpy error", _name);
}

void CudaImagePyramidHost::copyFromHost(int width, int height, cudaTextureFilterMode filter_mode, const void* source)
{
    initialize(width, height, filter_mode);
    copyFromHost(source);
}

void CudaImagePyramidHost::copyTo(CudaImagePyramidHost& target)
{
    assert(target._typeSize == _typeSize && target._textureType == _textureType);
    target.initialize(_baseWidth, _baseHeight, _filterMode, _numLayers);

    cudaMemcpyArrayToArray(target._storage,0,0,_storage,0,0,
                           _baseWidth*_baseHeight*_numLayers*_typeSize, cudaMemcpyDeviceToDevice);
    checkCUDAError("Memcpy error", _name);
}

void CudaImagePyramidHost::bindTexture()
{
    const textureReference* constTexRefPtr=NULL;
	const void* constTexRef = TexSynth::getTextureReferenceByName(_texture_name);
    cudaGetTextureReference(&constTexRefPtr, constTexRef);
    checkCUDAError("Can't get tex ref for bind", _texture_name);
    cudaChannelFormatDesc formatDesc = constTexRefPtr->channelDesc;

    cudaBindTextureToArray(constTexRefPtr, _storage, &formatDesc);
    checkCUDAError("Bind error", _name);
}

void CudaImagePyramidHost::unbindTexture()
{
    const textureReference* constTexRefPtr=NULL;
    const void* constTexRef = TexSynth::getTextureReferenceByName(_texture_name);
	cudaGetTextureReference(&constTexRefPtr, constTexRef);
    checkCUDAError("Can't get tex ref for unbind", _texture_name);

    cudaUnbindTexture(constTexRefPtr);
    checkCUDAError("Unbind error", _name);
}

