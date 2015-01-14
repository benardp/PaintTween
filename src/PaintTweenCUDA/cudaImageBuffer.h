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

#ifndef _CUDA_IMAGE_BUFFER_
#define _CUDA_IMAGE_BUFFER_

#include "cudaHostUtil.h"
#include "types.h"

#include <cmath>
#include <cstdio>
#include <assert.h>

#include <QtCore/QList>
#include <QtCore/QVector>

#include <cuda_runtime.h>

// Code to encapsulate a single level image buffer for cuda.
//
// There are two halves: *Host and *Device. Kernel and
// device functions can only call the *Device type, while
// memory allocation and management can only be done by the
// *Host type. The *Host half is defined outside the kernel 
// or device functions. When a kernel is called, the *Host 
// is converted to a *Device via deviceType() automatically 
// by an operator. 
//
// Unlike CudaImagePyramid, CudaImageBuffer is templated
// so the same self-managing memory pattern doesn't easily
// apply, and the buffers must be freed individually from the
// control code.

//#define CUDA_IMAGE_BUFFER_DEVICE_DEBUG

template <class T>
class CudaImageBufferDevice
{
    public:
        __host__ __device__ CudaImageBufferDevice(T* storage, int pitch, int width, int height) : 
            _storage(storage), _pitch(pitch), _width(width), _height(height) {}

#ifdef CUDA_IMAGE_BUFFER_DEVICE_DEBUG
        __host__ __device__ T pixel(int x, int y, int line = 0)
        {
            if (x < 0 || x >= _width || y < 0 || y >= _height) {
                printf("Out of bounds access line %d, %d %d (%d %d)\n", line, x, y, _width, _height);
                return T();
            }

#else
        __host__ __device__ T pixel(int x, int y)
        {
#endif
            return *((T*)((char*)_storage + y * _pitch) + x);
        }

        __host__ __device__ T pixelWrap(int x, int y)
        {
            int wx = x - _width*floorf((float)x/_width);
            int wy = y - _height*floorf((float)y/_height);

            return *((T*)((char*)_storage + wy * _pitch) + wx);
        }

#ifdef CUDA_IMAGE_BUFFER_DEVICE_DEBUG
        __host__ __device__ void setPixel(const T& value, int x, int y, int line = 0)
        {
            if (x < 0 || x >= _width || y < 0 || y >= _height) {
                printf("Out of bounds access line %d, %d %d (%d %d)\n", line, x, y, _width, _height);
                return;
            }
#else
        __host__ __device__ void setPixel(const T& value, int x, int y)
        {
#endif
            T* p = (T*)((char*)_storage + y * _pitch) + x; *p = value;
        }
	__host__ __device__ T at (int i, int level) {
            int y = floorf((float)i / (float)width(level));
            int x = (i - y*width(level));
	    return pixel(x,y);
        }

        __host__ __device__ void atomicAddition(const T& value, float x, float y)
        {
            // bilinearly interpolation
	    float wx = x - floorf(x);
	    float wy = y - floorf(y);

	    T* p = (T*)((char*)_storage + (int)floorf(y) * _pitch) + (int)floorf(x);
	    atomicAdd(p, value*(1.0-wy)*(1.0-wx));
	    p = (T*)((char*)_storage + (int)ceilf(y) * _pitch) + (int)floorf(x);
	    atomicAdd(p, value*(1.0-wy)*wx);
	    p = (T*)((char*)_storage + (int)ceilf(y) * _pitch) + (int)ceilf(x);
	    atomicAdd(p, value*wy*wx);
	    p = (T*)((char*)_storage + (int)floorf(y) * _pitch) + (int)ceilf(x);
	    atomicAdd(p, value*wy*(1.0-wx));
        }

        __host__ __device__ bool inBounds(int x, int y, int level, int border = 0) {
            return x >= border && y >= border && x < width(level) - border && y < height(level) - border;
        }

        __host__ __device__ int width(int level = 0) { return _width >> level; }
        __host__ __device__ int height(int level = 0) { return _height >> level; }

    protected:
        T*  _storage;
        int _pitch;
        int _width, _height; 

};

template <class T>
class CudaImageBufferHost
{
    public:
        CudaImageBufferHost() : _storage(NULL), _pitch(0), _width(0), _height(0) {}
         // This class doesn't count references, so it can't allow copying.
        CudaImageBufferHost(const CudaImageBufferHost<T>& copy) { assert(0); }
        ~CudaImageBufferHost() { clear(); }

	void initialize(int width, int height)
        {
            if (isInitialized() && width == _width && height == _height) {
                return;
            }
            clear();

            _width = width;
            _height = height;
            int width_in_bytes = _width*sizeof(T);

            cudaMallocPitch(&_storage, &_pitch, width_in_bytes, _height);
            checkCUDAError("Failure to allocate image buffer");
            cudaMemset2D(_storage, _pitch, 0, width_in_bytes, _height);
            checkCUDAError("Failure to memset image buffer");

            _host_cache_dirty = true;
        }

        void reset()
        {
            if (!isInitialized())
                return;

            int width_in_bytes = _width*sizeof(T);
            cudaMemset2D(_storage, _pitch, 0, width_in_bytes, _height);
            checkCUDAError("Failure to memset image buffer");
            _host_cache_dirty = true;
        }

        void clear() 
        {
            if (!isInitialized())
                return;

            cudaFree(_storage);
            checkCUDAError("Free error on image buffer");
            _storage = NULL;
            _width = 0;
            _height = 0;
            clearHistory();
        }

	void copyFromHost(const QVector<T>& source, int level)
        {
            int size_needed = width(level)*height(level);
            if ((int)(source.size()) != size_needed) {
                printf("copyFromHost: needed %d bytes, got %d\n", size_needed, (int)(source.size()));
                assert(0);
            }
            int src_pitch = width(level)*sizeof(T);
	    cudaMemcpy2D(_storage, _pitch, source.data(), src_pitch,
                         width(level)*sizeof(T), height(level), cudaMemcpyHostToDevice); 
            checkCUDAError("memcpy error from host to image buffer");

            _host_cache_dirty = true;
        }

	void copyToHost(QVector<T>& destination, int level)
        {
            // Note: resizes destination as necessary.
            int size_needed = width(level)*height(level);
            destination.resize(size_needed);
            int dest_pitch = width(level)*sizeof(T);
	    cudaMemcpy2D(destination.data(), dest_pitch, _storage, _pitch,
                         width(level)*sizeof(T), height(level), cudaMemcpyDeviceToHost); 
            checkCUDAError("memcpy error from image buffer to host");
        }
        
        void copyTo(CudaImageBufferHost& target)
        {
            assert(target._width == _width && target._height == _height && target._pitch == _pitch);
            cudaMemcpy2D(target._storage, target._pitch, _storage, _pitch,
                         _width*sizeof(T), _height, cudaMemcpyDeviceToDevice);
            checkCUDAError("memcpy error for image buffer");
            target._host_cache_dirty = true;
        }

        void copyTo(cudaArray* target, int level)
        {
            cudaMemcpy2DToArray(target, 0, 0, _storage, _pitch, width(level)*sizeof(T),
                height(level), cudaMemcpyDeviceToDevice);
            checkCUDAError("memcpy error for cuda array");
        }

        void copyTo(cudaArray* target, int level, int numStyles)
        {
            cudaMemcpy2DToArray(target, 0, 0, _storage, _pitch, width(level)*sizeof(T),
                height(level)/numStyles, cudaMemcpyDeviceToDevice);
            checkCUDAError("memcpy error for cuda array");
        }

        void copyFrom(cudaArray* target, int level)
        {
            cudaMemcpy2DFromArray(_storage, _pitch, target, 0, 0, width(level)*sizeof(T),
                height(level), cudaMemcpyDeviceToDevice);
            checkCUDAError("memcpy error for cuda array");
            _host_cache_dirty = true;
        }

        const T& pixel(int x, int y)
        {
            updateHostCache();
            return _host_cache[y*_width + x];
        }

        void clearHistory() { _history.clear(); }
        int historySize() const { return (int)(_history.size()); }
        void copyCurrentToHistory()
        {
            updateHostCache();
            _history.push_front(_host_cache);
            if(_history.size() > maxSizeHistory)
                _history.takeLast();
        }

        void copyHistoryToCurrent(int position)
        {
            assert(position >= 0 && position < (int)(_history.size()));
            copyFromHost(_history[position], 0);
        }

        bool isInitialized() const { return _storage != NULL; }

        int width(int level = 0) const { return _width >> level; }
        int height(int level = 0) const { return _height >> level; }
        size_t pitch() const {return _pitch; }
	const QVector<T>& hostCache() { updateHostCache(); return _host_cache; }

        dim3 blockCounts(int level, dim3 threadsPerBlock) {
            return dim3( ( (width(level)+threadsPerBlock.x-1) / threadsPerBlock.x),
                     ((height(level)+threadsPerBlock.y-1) / threadsPerBlock.y));
        }

        CudaImageBufferDevice<T> deviceType()
        {
            // The assumption here is that when we call deviceType, we are passing the buffer
            // to cuda for writing. So we can safely make the host cache dirty.
            _host_cache_dirty = true;
            return CudaImageBufferDevice<T>(_storage, _pitch, _width, _height);
        }
        operator CudaImageBufferDevice<T>() { return deviceType(); }

        static const int maxSizeHistory = 15;

    protected:
        void updateHostCache()
        {
            if (_host_cache_dirty) {
                copyToHost(_host_cache, 0);
            }
            _host_cache_dirty = false;
        }

    protected:
        // _storage is a pointer into device memory, i.e. a handle for cuda.
        T*  _storage;
        size_t _pitch;
        int _width, _height;
        bool _host_cache_dirty;
	QVector<T> _host_cache;
	QList< QVector< T > > _history;
};

template <class T>
class CudaBufferPair
{
public:
    ~CudaBufferPair() { clear(); }

    void initialize(int width, int height) {
        _base.initialize(width, height);
        _working.initialize(width,height);
    }

    void reset() {
        _base.reset();
        _working.reset();
    }

    void clear() {
        _base.clear();
        _working.clear();
    }

    void copyWorkingToBase() {
        _working.copyTo(_base);
    }

    void copyBaseToWorking() {
        _base.copyTo(_working);
    }

    void clearHistory() { _base.clearHistory(); _working.clearHistory(); }
    int historySize() const { return _base.historySize(); }
    void copyCurrentToHistory()
    {
        _base.copyCurrentToHistory();
        _working.copyCurrentToHistory();
    }

    void copyHistoryToCurrent(int position)
    {
        _base.copyHistoryToCurrent(position);
        _working.copyHistoryToCurrent(position);
    }

    int width(int level=0) const { return _base.width(level); }
    int height(int level=0) const { return _base.height(level); }

    bool isInitialized() const { return _base.isInitialized(); }
    dim3 blockCounts(int level, dim3 threadsPerBlock) { return _base.blockCounts(level, threadsPerBlock); }

    CudaImageBufferHost<T>& base() { return _base; }
    CudaImageBufferHost<T>& working() { return _working; }

    // Automatic conversion uses the base buffer.
    operator CudaImageBufferDevice<T>() { return _base.deviceType(); }

protected:
    CudaImageBufferHost<T> _base;
    CudaImageBufferHost<T> _working;
};

// Specialized copying functions for dealing with color pyramids. Could probably be member functions
// with template specialization.

template <class T>
void copyToHostColor4(CudaImageBufferHost<T>& src, QVector<Color4>& colors, int level)
{
    QVector<T> local;
    src.copyToHost(local, level);
    colors.resize(local.size());
    for (int i = 0; i < (int)(local.size()); i++) {
        colors[i] = local[i].toColor4();
    }
}

#endif

