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

#include "cudaTexture.h"

#include <cstdio>
#include <cstring>

#include "texSynth_kernel.h"

std::vector<CudaTextureHost*> CudaTextureHost::_instances;

CudaTextureHost::CudaTextureHost(int textureType, int typeSize, const char* name) :
    _storage(NULL), _typeSize(typeSize), _textureType(textureType),
    _width(0), _height(0), _in_destructor(false)
{
    strncpy(_name, name, 200);
    // By convention, the associated Cuda texture is (name)_TEXTURE
    snprintf(_texture_name, 200, "%s_TEXTURE", _name);
}

CudaTextureHost::~CudaTextureHost()
{
    _in_destructor = true;
    clear();
}

void CudaTextureHost::initialize(int width, int height, cudaTextureFilterMode filter_mode)
{
    if (isInitialized() && width == _width && height == _height && filter_mode == _filterMode) {
	return;
    }

    clear();

    _width = width;
    _height = height;
    _filterMode = filter_mode;

    // Get the texture and its channel descriptor to allocate the storage.
    const textureReference* constTexRefPtr=NULL;
	const void* constTexRef = TexSynth::getTextureReferenceByName(_texture_name);
    cudaGetTextureReference(&constTexRefPtr, constTexRef );
    checkCUDAError("Can't get tex ref for init", _texture_name);
    cudaChannelFormatDesc formatDesc = constTexRefPtr->channelDesc;

    cudaMallocArray(&_storage, &formatDesc, _width, _height);
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
    texRefPtr->normalized = false; // Use unnormalized (pixel) coordinates for addressing.

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

void CudaTextureHost::clear()
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
    _width = 0; _height = 0;
}

void CudaTextureHost::clearAllInstances()
{
    for (size_t i = 0; i < _instances.size(); i++) {
	_instances[i]->clear();
    }
}

void CudaTextureHost::copyFromHost(const void* source)
{
    assert(isInitialized());

    cudaMemcpyToArray(_storage, 0,0, source, _width*_height*_typeSize, cudaMemcpyHostToDevice);
    checkCUDAError("Memcpy error", _name);
}

void CudaTextureHost::copyFromHost(int width, int height, cudaTextureFilterMode filter_mode, const void* source)
{
    initialize(width, height, filter_mode);
    copyFromHost(source);
}

void CudaTextureHost::copyTo(CudaTextureHost& target)
{
    assert(target._typeSize == _typeSize);
    target.initialize(_width, _height, _filterMode);

    cudaMemcpyArrayToArray(target._storage,0,0,_storage,0,0,
			   _width*_height*_typeSize, cudaMemcpyDeviceToDevice);
    checkCUDAError("Memcpy error", _name);
}

void CudaTextureHost::bindTexture()
{
    const textureReference* constTexRefPtr=NULL;
    const void* constTexRef = TexSynth::getTextureReferenceByName(_texture_name);
	cudaGetTextureReference(&constTexRefPtr, constTexRef);
    checkCUDAError("Can't get tex ref for bind", _texture_name);
    cudaChannelFormatDesc formatDesc = constTexRefPtr->channelDesc;

    cudaBindTextureToArray(constTexRefPtr, _storage, &formatDesc);
    checkCUDAError("Bind error", _name);
}

void CudaTextureHost::unbindTexture()
{
    const textureReference* constTexRefPtr=NULL;
	const void* constTexRef = TexSynth::getTextureReferenceByName(_texture_name);
    cudaGetTextureReference(&constTexRefPtr, constTexRef);
    checkCUDAError("Can't get tex ref for unbind", _texture_name);

    cudaUnbindTexture(constTexRefPtr);
    checkCUDAError("Unbind error", _name);
}

