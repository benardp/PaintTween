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

#ifndef CUDATEXTURE_H
#define CUDATEXTURE_H

#include <vector>

class CudaTextureHost
{  
public:
    CudaTextureHost(int textureType, int typeSize, const char* name);
    ~CudaTextureHost();

    void initialize(int width, int height, cudaTextureFilterMode filter_mode);

    void clear();
    static void clearAllInstances();

    void copyFromHost(const void* source);
    void copyFromHost(int width, int height, cudaTextureFilterMode filter_mode, const void* source);
    void copyTo(CudaTextureHost& target);

    bool isInitialized() const { return _storage != NULL; }

    int width() { return _width; }
    int height() { return _height; }

protected:
    void bindTexture();
    void unbindTexture();

protected:
    cudaArray* _storage;
    int _typeSize;
    int _textureType;

    int _width, _height;
    cudaTextureFilterMode _filterMode;

    char _name[200];
    char _texture_name[200];

    bool _in_destructor;

    static std::vector<CudaTextureHost*> _instances;
};


#define CUDA_TEXTURE(TYPE,DATA_TYPE,NAME) \
	    CudaTextureHost NAME((TYPE),sizeof(DATA_TYPE), #NAME); \
	    texture<DATA_TYPE,TYPE> NAME ## _TEXTURE; \

#define TEXTURE_FETCH1D(NAME,X) tex1D(NAME ## _TEXTURE,(X))
#define TEXTURE_FETCH2D(NAME,X,Y) tex1D(NAME ## _TEXTURE,(X),(Y))
#define TEXTURE_FETCH3D(NAME,X,Y,Z) tex1D(NAME ## _TEXTURE,(X),(Y),(Z))

#endif // CUDATEXTURE_H
