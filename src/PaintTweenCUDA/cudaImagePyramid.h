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

#ifndef _CUDA_IMAGE_PYRAMID_
#define _CUDA_IMAGE_PYRAMID_

#include <vector>

// Code to encapsulate as best we can the image pyramid datastructure
// used on the cuda side. 
//
// Super cumbersome because you can't pass texture<> references as function arguments,
// so you are forced to use macros to bundle the host and device calls together.
//
// An image pyramid is a bundle of two things: a CudaImagePyramidHost,
// and a texture<> reference. The CUDA_IMAGE_PYRAMID macro defines these things together.
// The macro must be defined at file scope (global scope) in the .cu file.
// A naming convention is used for the bundle: given NAME, the
// CudaImagePyramidHost is called NAME, the texture is called NAME_TEXTURE_PYRAMID.
//  
// Texture fetches from the pyramid must be done with the PYRAMID_FETCH
// macro, which uses the naming convention to find the proper texture
// and offsets array.
//
// Note that this class is called a pyramid only for historical reasons at this point:
// it does not actually store multiple levels of an image pyramid at the same time.
// The different levels are loaded as required from the host side using copyFromHost.
//
// Historically, it made sense to store the full pyramid on the CUDA side when doing
// synthesis for a single frame. When doing animation synthesis, there is less use in
// storing the entire pyramid, because animation generally proceeds frame-to-frame for
// each level, so the pyramid is entirely flushed at each frame.

class CudaImagePyramidHost
{
    public:
        CudaImagePyramidHost(int textureType, int typeSize, const char* name);
        ~CudaImagePyramidHost();

        void initialize(int width, int height, cudaTextureFilterMode filter_mode, int depth = 1);

        void clear();
        static void clearAllInstances();

        void copyFromHost(const void* source);
        void copyFromHost(int width, int height, const void* source, int layer);
        void copyFromHost(int width, int height, cudaTextureFilterMode filter_mode, const void* source);
        void copyTo(CudaImagePyramidHost& target);

        bool isInitialized() const { return _storage != NULL; }

        int width() { return _baseWidth; }
        int height() { return _baseHeight; }
    
    protected:
        void bindTexture();
        void unbindTexture();

    protected:
        cudaArray* _storage;
        int _typeSize;
        int _textureType;

	int _baseWidth, _baseHeight, _numLayers;
        cudaTextureFilterMode _filterMode;

        char _name[200];
        char _texture_name[200];

        bool _in_destructor;

        static std::vector<CudaImagePyramidHost*> _instances;
};

#define CUDA_IMAGE_PYRAMID(TYPE,NAME) \
    CudaImagePyramidHost NAME(cudaTextureType2D, sizeof(TYPE), #NAME); \
    texture<TYPE, cudaTextureType2D> NAME ## _TEXTURE_PYRAMID;
#define PYRAMID_FETCH(NAME,X,Y) tex2D(NAME ## _TEXTURE_PYRAMID, (X) + 0.5, (Y) + 0.5)

#define CUDA_IMAGE_PYRAMID_LAYERED(TYPE,NAME) \
    CudaImagePyramidHost NAME(cudaTextureType2DLayered, sizeof(TYPE), #NAME); \
    texture<TYPE, cudaTextureType2DLayered> NAME ## _TEXTURE_PYRAMID;

#define PYRAMID_FETCH_LAYER(NAME,X,Y,LAYER) tex2DLayered(NAME ## _TEXTURE_PYRAMID, (X) + 0.5, (Y) + 0.5,(LAYER))

#endif

