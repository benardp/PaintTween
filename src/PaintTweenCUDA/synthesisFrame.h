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

#ifndef SYNTHESISFRAME_H
#define SYNTHESISFRAME_H

#include "types.h"
#include "imagePyramid.h"
#include "dataAccess.h"

class Style;
class WorkingBuffers;
class SynthesisThreadData;

class SynthesisFrame
{
public:
    static const int INIT_SPATIAL_UPSAMPLE = 0x1;
    static const int INIT_TEMPORAL_UPSAMPLE = 0x2;
    static const int INIT_PREVIOUS_PASS = 0x4;

public:
    void initialize(int frame_number, const Style* style, WorkingBuffers* buffers, SynthesisThreadData* thread_state = NULL);

    static void setSynthesisRange(int first_frame, int last_frame,
                                  int num_levels, bool do_keyframe_interpolation,
                                  bool do_caching);

    const ImagePyramid<Color4>& imageBase() const { return _imageBase; }
    const ImagePyramid<float>& imageOrientation() const { return _imageOrientation; }
    const ImagePyramid<float>& imageDistanceTransform() const { return _imageDistanceTransform; }
    const ImagePyramid<float2>& imageScale() const { return _imageScale; }
    const ImagePyramid<Color4>& imageRandom() const { return _imageRandom; }
    const ImagePyramid<int>& imageId() const { return _imageId; }
    const ImagePyramid<float2>& imageSurfaceId() const { return _imageSurfaceId; }
    const ImagePyramid<float2>& imageVelB() const { return _imageVelB; }
    const ImagePyramid<float2>& imageVelF() const { return _imageVelF; }

    const ImagePyramid<Color4>& forwardGuide() const { return _forwardGuide; }
    const ImagePyramid<Color4>& backwardGuide() const { return _backwardGuide; }

    const ImagePyramid<Color4>& frameToKeyRibbonB() const { return _frameToKeyRibbonB; }
    const ImagePyramid<Color4>& frameToKeyRibbonF() const { return _frameToKeyRibbonF; }

    void createRandomPyramid(int level);

    void initializeGuide(const QVector<Color4> &data, int width, int height, bool timeIsForwards, bool writeToOutput);
    void initializeFrameToKeyRibbon(const QVector<RibbonP>& data, int width, int height, bool timeIsForwards, bool writeToOutput);
    bool loadGuide(bool timeIsForwards);
    bool loadFrameToKeyRibbon(bool timeIsForwards);

    bool synthesizeConsecutive(int level, int pass, int time_step, bool is_first_frame, bool time_is_forwards, int source_pass, bool firstPass);
    bool synthesizeInterpolated(int level, int this_pass, int source_pass, int time_step, SynthesisFrame* prior_frame, SynthesisFrame* next_frame);
    bool refine(int level, int this_pass, int last_coarse_pass, int time_step, SynthesisFrame* prior_frame, SynthesisFrame* next_frame, int op_flags);
    bool spatialUpsample(int coarse_level, int fine_level, int source_pass, bool save_images = true);

    void mergeRandomly(int level, bool time_is_forwards, bool force_valid=false);

    bool loadImages(int level);
    bool loadAnimImages(int level, int time_step);
    bool loadRibbonPair(int level, int previous_frame);
    void cachePyramids(int level, bool overwrite = false);

    bool saveImages(int level, int pass, int version = 0);
    bool loadLastSavedOffsets(int level);
    bool linkImagesToPass(int level, int pass);
    int lastSavedPass(int level) { return _last_saved_pass[level]; }

    int frameNumber() const { return _frame_number; }

protected:
    bool initImages();

    bool propagateAndScatter(int level, int iterations, int keyIndex = -1);

    bool advectNeighbors(int level, int source_pass, int time_step, SynthesisFrame* prior_frame, SynthesisFrame* next_frame, int op_flags = 0);

    void updateParameters(PassDirection direction);

    void cacheAnimPyramids(int level, bool overwrite = false);
    void clearAnimPyramids();

protected:
    int _frame_number;
    const Style* _style;
    WorkingBuffers* _working_buffers;

    static int _first_frame, _last_frame, _num_levels;
    static bool _do_keyframe_interpolation;
    static bool _do_caching;

    ImagePyramid<Color4>  _imageBase;
    ImagePyramid<float>   _imageOrientation;
    ImagePyramid<float>   _imageDistanceTransform;
    ImagePyramid<float2>   _imageScale;
    ImagePyramid<Color4>  _imageRandom;
    ImagePyramid<int>     _imageId;
    ImagePyramid<float2>  _imageSurfaceId;
    ImagePyramid<float2>  _imageVelB;
    ImagePyramid<float2>  _imageVelF;
    int _anim_image_time_step;

    ImagePyramid<Color4>  _forwardGuide;
    ImagePyramid<Color4>  _backwardGuide;

    ImagePyramid<Color4>  _frameToKeyRibbonB;
    ImagePyramid<Color4>  _frameToKeyRibbonF;

    SynthesisThreadData* _thread_state;

    QVector<int> _last_saved_pass;
};

#endif
