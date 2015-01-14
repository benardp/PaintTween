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

#include "cudaHostUtil.h"

#if _MSC_VER
#include <Windows.h>
#endif

#include <QtGlobal>
#ifdef Q_OS_MAC
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <QtCore/QDebug>

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        qCritical("Cuda error %s: %s\n", msg,
                  cudaGetErrorString( err));
        exit(EXIT_FAILURE);
    }
}

void checkCUDAError(const char *msg, const char *name)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        qCritical("Cuda error: %s for %s: %s", msg,
                  name, cudaGetErrorString( err));
        exit(EXIT_FAILURE);
    }
}

void reportGLError(const char* msg)
{
    GLint error = glGetError();
    if (error != 0) {
        qCritical("GL Error %s: %s\n", msg, gluErrorString(error));
    }
}

void renderTexturedQuad(float sBound, float tBound)
{
    glBegin(GL_QUADS);

    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(sBound, 0);
    glVertex2f(1, 0);
    glTexCoord2f(sBound, tBound);
    glVertex2f(1.f, 1);
    glTexCoord2f(0, tBound);
    glVertex2f(0, 1);

    glEnd();
}

void renderTexturedQuad(float tBound) {
    renderTexturedQuad(tBound,tBound);
}
