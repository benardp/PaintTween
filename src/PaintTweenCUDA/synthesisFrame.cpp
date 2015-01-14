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

#include "synthesisFrame.h"
#include "synthesisProcessor.h"
#include "imageIO.h"
#include "stats.h"

#include "texSynth_kernel.h"

#include <QtCore/QDebug>

int SynthesisFrame::_first_frame = 0;
int SynthesisFrame::_last_frame = 0;
int SynthesisFrame::_num_levels = 0;
bool SynthesisFrame::_do_keyframe_interpolation = false;
bool SynthesisFrame::_do_caching = false;

#define WAIT_AND_RETURN_IF_TERMINATED(msg) if (_thread_state) { _thread_state->waitForContinue(msg); if (_thread_state->_synthesisState == STATE_TERMINATING) return false; }

void SynthesisFrame::initialize(int frame_number, const Style* style, WorkingBuffers* buffers, SynthesisThreadData* thread_state)

{
    _frame_number = frame_number;
    _style = style;
    _working_buffers = buffers;
    _thread_state = thread_state;
    _anim_image_time_step = 0;
}

void SynthesisFrame::setSynthesisRange(int first_frame, int last_frame,
                                       int num_levels, bool do_keyframe_interpolation,
                                       bool do_caching)
{
    _first_frame = first_frame;
    _last_frame = last_frame;
    _num_levels = num_levels;
    _do_keyframe_interpolation = do_keyframe_interpolation;
    _do_caching = do_caching;
}

void SynthesisFrame::createRandomPyramid(int level)
{
    DataAccess& data = DataAccess::instance();

    // Make each pyramid repeatable.
    srand(_frame_number*level);

    if(!_imageRandom.isInitialized())
    {
        initImages();

        int imageWidth = _imageBase.width();
        int imageHeight = _imageBase.height();
        _imageRandom.initialize(imageWidth, imageHeight, Color4(), ImageIO::parsePath(data.getTemporaryDir() + "/randomPyramid_%2.%1.exr",_frame_number),
                                       LayerDesc("random","random",QStringList()<<"x"<<"y"<<"theta"<<"style"));
    }

    if (!_imageRandom.load(level))
    {
        if(!_imageOrientation.isLoaded(level))
            _imageOrientation.load(level);

        int numStyles = data.getNumStyles();

        int imageWidth = _imageBase.width(level);
        int imageHeight = _imageBase.height(level);
        bool transparencyAllowed = data.getBoolParameter("transparencyAllowed");
        bool orientationEnabled = data.getBoolParameter("orientationEnabled");
        bool setupWithKeys = _do_keyframe_interpolation && (_style->keyFrameSubrange().size() > 0);

        if(setupWithKeys){
            numStyles = _style->keyFrameSubrange().size();
        }

        assert(numStyles != 0 );

        const int MAX_TRIALS = 50;

        // Set base level first.
        for(int y=0; y < imageHeight; y++){
            for(int x=0; x < imageWidth; x++){
                int thisStyle = 0;
                if(setupWithKeys){
                    thisStyle = rand() % numStyles + _style->firstKeyFrameFullIndex();
                }

                int offset_x, offset_y;

                int trial_counter = 0;
                do {
                    offset_x = rand() % ((data.getStyleWidth(thisStyle) >> level) - 1);
                    offset_y = rand() % ((data.getStyleHeight(thisStyle) >> level) - 1);

                    trial_counter++;

                    // This is the "no swiss cheese" check to make sure we start out
                    // every opaque area with an opaque guess. Probably should revisit this.
                    // -fcole sep 29 2011
                } while ((!transparencyAllowed) &&
                         (_imageBase.pixel(x,y,level).a > ALPHA_THRESHOLD) &&
                         (_style->output(thisStyle).pixel(offset_x, offset_y, level).a <ALPHA_THRESHOLD) &&
                         (trial_counter < MAX_TRIALS));

                float orientation = (orientationEnabled) ?
                            _imageOrientation.pixel(x,y,level) - _style->orientation(thisStyle).pixel(offset_x, offset_y, level) : 0.f;
                if(orientation>M_PI || orientation<-M_PI)
                    qCritical("Orientation pb");

                // Add some randomness after the decimal place as a helper for kernels that
                // need random floating point numbers.
                float rx = (float)rand() / (float)RAND_MAX;
                float ry = (float)rand() / (float)RAND_MAX;

                Color4 packed = Color4(offset_x + rx, offset_y + ry, orientation, thisStyle);

                _imageRandom.setPixel(packed, x, y, level);
            }
        }
        if (_do_caching) {
            _imageOrientation.cache(level);
        }
    }

    TexSynth::uploadImage_Color4(level, TS_LAYER_RANDOM, _imageRandom);
}

void SynthesisFrame::initializeGuide(const QVector<Color4>& data, int width, int height,
                                         bool timeIsForwards, bool writeToOutput)
{
    if (timeIsForwards) {
        _forwardGuide.initialize(data, width, height,
                                 ImageIO::parsePath(DataAccess::instance().getTemporaryDir()+"/forwardGuidePyramid_%2.%1.exr",_frame_number),
                                 LayerDesc("rgba","forwardGuide",QStringList()<<"R"<<"G"<<"B"<<"A"));
    } else {
        _backwardGuide.initialize(data, width, height,
                                  ImageIO::parsePath(DataAccess::instance().getTemporaryDir()+"/backwardGuidePyramid_%2.%1.exr",_frame_number),
                                  LayerDesc("rgba","backwardGuide",QStringList()<<"R"<<"G"<<"B"<<"A"));
    }

    if (writeToOutput) {
        QString cachename = ImageIO::parsePath(QString("%1/%2Guide.%3.exr").arg(DataAccess::instance().getOutDir()).arg((timeIsForwards) ? "forward" : "backward"),_frame_number);
        ImageIO::writeImage(data, cachename, width, height);
    }
}

void SynthesisFrame::initializeFrameToKeyRibbon(const QVector<RibbonP>& data, int width, int height,
                                                    bool timeIsForwards, bool writeToOutput)
{
    if (timeIsForwards) {
        initializeRibbonPyramid(data, _frameToKeyRibbonB, width, height,
                                ImageIO::parsePath(DataAccess::instance().getTemporaryDir()+"/frameToKeyRibbonB_%2.%1.exr",_frame_number));
    } else {
        initializeRibbonPyramid(data, _frameToKeyRibbonF, width, height,
                                ImageIO::parsePath(DataAccess::instance().getTemporaryDir()+"/frameToKeyRibbonF_%2.%1.exr",_frame_number));
    }

    if (writeToOutput) {
        QString cachename = ImageIO::parsePath(QString("%1/frameToKeyRibbon%2.%3.exr").arg(DataAccess::instance().getOutDir()).arg((timeIsForwards) ? 'B' : 'F'),_frame_number);
        ImageIO::writeImage(data, cachename, width, height);
    }
}


bool SynthesisFrame::loadGuide(bool timeIsForwards)
{
    QVector<Color4> guide;
    int width, height;
    QString cachename = ImageIO::parsePath(QString("%1/%2Guide.%3.exr").arg(DataAccess::instance().getOutDir()).arg((timeIsForwards) ? "forward" : "backward"),_frame_number);
    bool success = ImageIO::readImage(cachename, guide, width, height);
    if (success) {
        initializeGuide(guide, width, height, timeIsForwards, false);
    }
    return success;
}

bool SynthesisFrame::loadFrameToKeyRibbon(bool timeIsForwards)
{
    QVector<RibbonP> ribbon;
    int width, height;
    QString cachename = ImageIO::parsePath(QString("%1/frameToKeyRibbon%2.%3.exr").arg(DataAccess::instance().getOutDir()).arg((timeIsForwards) ? 'B' : 'F'),_frame_number);
    bool success = ImageIO::readImage(cachename, ribbon, width, height);
    if (success) {
        initializeFrameToKeyRibbon(ribbon, width, height, timeIsForwards, false);
    }
    return success;
}


bool SynthesisFrame::synthesizeConsecutive(int level, int pass, int time_step,
                                                bool is_first_frame, bool time_is_forwards,
                                                int source_pass, bool firstPass)
{
    DataAccess& data = DataAccess::instance();
    updateParameters(time_is_forwards ? FORWARD : BACKWARD);

    int advect_ops = (_do_keyframe_interpolation) ? TexSynth::ADVECT_CHECK_FRAME_TO_KEY : 0;
    if(data.getSynthesisScheme() != S_FMBM_RM)
           advect_ops = advect_ops | TexSynth::ADVECT_UPDATE_RESIDUAL;

    if(firstPass){
        createRandomPyramid(level);

        if(is_first_frame){
            // Load some initial offsets for the first frame, either
            // random (if coarsest level), or upsampled from a coarser level

            if(level == _num_levels-1){
                // We are on the coarsest level, so initialize the buffer.
                // If we are using keyframes, look forward as best we can. Otherwise,
                // just randomize.
                if (_do_keyframe_interpolation) {
                    TexSynth::offsetsFromFrameToKeyRibbon(_working_buffers, level, time_is_forwards);
                } else {
                    TexSynth::randomizeOffsets(_working_buffers, level);
                }
            }else{
                // We are not on a higher level, so use offsets
                // from last level:
                _working_buffers->loadOffsets(level+1, _frame_number, source_pass);
                //WAIT_AND_RETURN_IF_TERMINATED("before upsample");
                TexSynth::upsample(_working_buffers, level);
                TexSynth::syncBuffers(_working_buffers, level);
            }
        }

        // Pause here to allow viewing the image in the main thread.
        WAIT_AND_RETURN_IF_TERMINATED("before advection");
        //saveImages(level, pass, 3);

        if (!is_first_frame && level == _num_levels-1){
            TexSynth::advectOffsets(_working_buffers, level, time_step, time_is_forwards, advect_ops);
            // This is a bit of a hack. Advect offsets leaves the result in base(), but we need it in
            // working so that the syncBuffers call below doesn't immediately overwrite it.
            _working_buffers->offsets.copyBaseToWorking();
            //saveImages(level, pass, 2);

        } else if (!is_first_frame && level < _num_levels-1){
            // Advect and then merge with the upsampled level
            TexSynth::advectOffsets(_working_buffers, level, time_step, time_is_forwards, advect_ops | TexSynth::ADVECT_IGNORE_RANDOM);

            //WAIT_AND_RETURN_IF_TERMINATED("after advection");
            //saveImages(level, pass, 2);

            _working_buffers->loadOffsets(level+1, _frame_number, source_pass);
            _working_buffers->loadResidual(level+1, _frame_number, source_pass);
            TexSynth::upsample(_working_buffers, level);

            TexSynth::syncBuffers(_working_buffers, level);

            // Pause before propagation and scatter to visualize the merged image
            //WAIT_AND_RETURN_IF_TERMINATED("after upsample");

            if (data.getSynthesisScheme() == S_FMBM_RM) {
                mergeRandomly(level, time_is_forwards);
            } else {
                TexSynth::mergeUsingResidual(_working_buffers, level, time_is_forwards);
            }
        }
    }else{
        TexSynth::advectOffsets(_working_buffers, level, time_step, time_is_forwards, advect_ops | TexSynth::ADVECT_IGNORE_RANDOM);

        //WAIT_AND_RETURN_IF_TERMINATED("after advect");
        //saveImages(level, pass, 2);

        _working_buffers->loadOffsets(level, _frame_number, pass-1);
        _working_buffers->loadResidual(level, _frame_number, pass-1);
        TexSynth::syncBuffers(_working_buffers, level,
                              TexSynth::DONT_COPY_OFFSETS | TexSynth::DONT_UPDATE_RESIDUAL);

        //WAIT_AND_RETURN_IF_TERMINATED("after load offsest");

        if (data.getSynthesisScheme() == S_FMBM_RM) {
            mergeRandomly(level, time_is_forwards);
        } else {
            TexSynth::mergeUsingResidual(_working_buffers, level, time_is_forwards);
        }
    }

    // Make sure residual is updated (could possibly be replaced with updateResiduals)
    TexSynth::syncBuffers(_working_buffers, level);

    // Pause before propagation and scatter to visualize the merged image
    WAIT_AND_RETURN_IF_TERMINATED("before propagate and scatter");
    //saveImages(level, pass, 1);

    int totalItr = data.getIntParameter("iterations");

    if (!propagateAndScatter(level, totalItr, data.keyFrameIndices().indexOf(_frame_number)))
        return false;

    saveImages(level, pass);

    WAIT_AND_RETURN_IF_TERMINATED("finished frame");

    cachePyramids(-1, (level == _num_levels-1));

    return true;
}

bool SynthesisFrame::advectNeighbors(int level, int source_pass, int time_step,
                                     SynthesisFrame* prior_frame, SynthesisFrame* next_frame,
                                     int op_flags)
{
    _working_buffers->resetAdvectedBuffers();

    if (_do_keyframe_interpolation) {
        op_flags |= TexSynth::ADVECT_CHECK_FRAME_TO_KEY;
    }

    if(_last_frame != _first_frame &&
            (_working_buffers->params().timeDerivativeInputWeight > 0.f
             || _working_buffers->params().timeDerivativeOutputWeight > 0.f
             || _working_buffers->params().temporalCoherenceWeight > 0.f) &&
            (prior_frame || next_frame || _do_keyframe_interpolation)) {

        if(prior_frame && source_pass >= 0) {
            prior_frame->loadImages(level);
            prior_frame->loadAnimImages(level, time_step);
            TexSynth::flipAnimBuffers();
            loadAnimImages(level, time_step);

            prior_frame->loadLastSavedOffsets(level);
            TexSynth::advectOffsets(_working_buffers, level, time_step, true, op_flags);

            prior_frame->cachePyramids(level);
        } else if (_do_keyframe_interpolation) {
            loadAnimImages(level, time_step);
            TexSynth::advectOffsets(_working_buffers, level, time_step, true,
                                    op_flags | TexSynth::ADVECT_DO_NOT_CHECK_PREVIOUS);
        }

        WAIT_AND_RETURN_IF_TERMINATED("after advect forward");

        if(next_frame && source_pass >= 0) {
            next_frame->loadImages(level);
            next_frame->loadAnimImages(level, time_step);
            TexSynth::flipAnimBuffers();
            loadAnimImages(level, time_step);

            next_frame->loadLastSavedOffsets(level);
            TexSynth::advectOffsets(_working_buffers, level, time_step, false, op_flags);

            next_frame->cachePyramids(level);
        } else if (_do_keyframe_interpolation) {
            loadAnimImages(level, time_step);
            TexSynth::advectOffsets(_working_buffers, level, time_step, false,
                                    op_flags | TexSynth::ADVECT_DO_NOT_CHECK_PREVIOUS);
        }

        WAIT_AND_RETURN_IF_TERMINATED("after advect backward");

    }else{
        loadAnimImages(level, time_step);
    }

    return true;
}

bool SynthesisFrame::synthesizeInterpolated(int level, int this_pass, int source_pass, int time_step,
                                            SynthesisFrame* prior_frame, SynthesisFrame* next_frame)
{
    DataAccess& data = DataAccess::instance();
    updateParameters(BIDIRECTIONAL);

    advectNeighbors(level, source_pass, time_step, prior_frame, next_frame);

    loadImages(level);

    _working_buffers->loadOffsets(level, _frame_number, source_pass);
    TexSynth::syncBuffers(_working_buffers, level, TexSynth::DONT_COPY_OFFSETS | TexSynth::DONT_UPDATE_RESIDUAL);

    WAIT_AND_RETURN_IF_TERMINATED("before propagate and scatter");

    /*if(_working_buffers->params().useGuide){
        QVector<int>::const_iterator it = qUpperBound(data.keyFrameIndices(),_frame_number);
        int nextKeyFrane = *it; it--;
        int prevKeyFrame = *it;
        _working_buffers->params().numFrames = nextKeyFrane - prevKeyFrame;
        _working_buffers->params().currentInterpolationIndex = (_frame_number - prevKeyFrame);
        TexSynth::uploadParameters(_working_buffers->params());
    }*/

    if (!propagateAndScatter(level, data.getIntParameter("iterations"), data.keyFrameIndices().indexOf(_frame_number)))
        return false;

    saveImages(level, this_pass);

    WAIT_AND_RETURN_IF_TERMINATED("finished frame");

    cachePyramids(level);

    return true;
}

bool SynthesisFrame::refine(int level, int this_pass, int last_coarse_pass, int time_step,
                            SynthesisFrame* prior_frame, SynthesisFrame* next_frame, int op_flags)
{
    DataAccess& data = DataAccess::instance();
    updateParameters(BIDIRECTIONAL);
    int last_fine_pass = this_pass - 1;

    if (!initImages())
        return false;

    if (!advectNeighbors(level, last_fine_pass, time_step, prior_frame, next_frame))
        return false;

    // Set up initial guess using appropriate upsampling mode.
    if (op_flags & INIT_SPATIAL_UPSAMPLE) {
        int coarse_level = level + 1;
        if (!spatialUpsample(coarse_level, level, last_coarse_pass, false))
            return false;
    } else if (op_flags & INIT_TEMPORAL_UPSAMPLE) {
        TexSynth::mergeAdvectedBuffers(_working_buffers, level, TexSynth::MERGE_FALLBACK_RANDOM);
    } else {
        loadLastSavedOffsets(level);
        //_working_buffers->loadOffsets(level, _frame_number, last_fine_pass);
        //TexSynth::mergeAdvectedBuffers(_working_buffers, level, TexSynth::MERGE_FALLBACK_LAST_PASS);
        TexSynth::syncBuffers(_working_buffers, level, TexSynth::DONT_COPY_OFFSETS | TexSynth::DONT_UPDATE_RESIDUAL);
    }

    // Reset the current image state and optimize.
    if (!loadImages(level))
        return false;

    WAIT_AND_RETURN_IF_TERMINATED("before propagate and scatter");

    if (!propagateAndScatter(level, data.getIntParameter("iterations"), data.keyFrameIndices().indexOf(_frame_number)))
        return false;

    saveImages(level, this_pass);

    WAIT_AND_RETURN_IF_TERMINATED("finished frame");

    cachePyramids(level);

    return true;
}

bool SynthesisFrame::spatialUpsample(int coarse_level, int fine_level, int source_pass, bool save_images)
{
    Q_UNUSED(source_pass);

    if (!loadLastSavedOffsets(coarse_level))
            return false;
    TexSynth::upsample(_working_buffers, fine_level);
    TexSynth::syncBuffers(_working_buffers, fine_level, TexSynth::DONT_COPY_OFFSETS | TexSynth::DONT_UPDATE_RESIDUAL);  

    if (save_images) {
        saveImages(fine_level, 0);
    }

    WAIT_AND_RETURN_IF_TERMINATED("after spatial upsample");

    return true;
}

bool SynthesisFrame::propagateAndScatter(int level, int iterations, int keyIndex)
{
    for(int itr = 0; itr < iterations; itr++){

        if(_do_keyframe_interpolation) {
            TexSynth::propagate(_working_buffers, level, 0, keyIndex);
        }

        for(int i = 128; i>1; i/=2){
            TexSynth::propagate(_working_buffers, level, i, keyIndex);
        }

        TexSynth::propagate(_working_buffers, level, 1, keyIndex);

        TexSynth::propagate(_working_buffers, level, 1, keyIndex);

        //WAIT_AND_RETURN_IF_TERMINATED("finish propagate");

        if(itr < iterations -1) {
            TexSynth::scatter(_working_buffers, level, keyIndex);
        }
        //WAIT_AND_RETURN_IF_TERMINATED("finish scatter");
    }

    return true;
}

void SynthesisFrame::mergeRandomly(int level, bool time_is_forwards, bool force_valid)
{
    float advectWeight = 0.5;

    if (_do_keyframe_interpolation) {
        // Perform randomized merge based on distance to keyframes.
        advectWeight = _style->keyFrameAdvectWeight(_frame_number, _first_frame, _last_frame, time_is_forwards);
    }

    TexSynth::mergeRandomly(_working_buffers, level, time_is_forwards, advectWeight, force_valid);
}

void SynthesisFrame::updateParameters(PassDirection direction)
{
    DataAccess& data = DataAccess::instance();

    TsParameters params = _working_buffers->params();
    if (_frame_number % 3 == 0) {
        params.advectionJitter = data.getFloatParameter("advectionJitter") * 0.01;
    } else {
        params.advectionJitter = 0;
    }
    params.direction = direction;
    if(_working_buffers->params().useGuide){
        QVector<int>::const_iterator it = qUpperBound(data.keyFrameIndices(),_frame_number);
        int nextKeyFrane = *it; it--;
        int prevKeyFrame = *it;
        params.numFrames = nextKeyFrane - prevKeyFrame;
        params.currentInterpolationIndex = (_frame_number - prevKeyFrame);
    }
    _working_buffers->setParams(params);
    TexSynth::uploadParameters(_working_buffers->params());
}

bool SynthesisFrame::initImages(){
    if(!_imageBase.isInitialized())
    {
        DataAccess& data = DataAccess::instance();
        bool found = data.getInputElement(IN_INPUT, _frame_number, _imageBase);
        if(!found) {
            qCritical()<<"Input image not found!";
            return false;
        }
        data.getInputElement(IN_ORIENTATION, _frame_number,_imageOrientation);
        data.getInputElement(IN_ID_MERGED, _frame_number, _imageId);
        int width, height;
        data.getImageDimensions(width,height);
        TexSynth::initialize_images(width,height);
    }
    return true;
}

bool SynthesisFrame::loadImages(int level){
    __TIME_CODE_BLOCK("SynthesisFrame::loadImages");

    if (!initImages()) {
        return false;
    }

    if(!_imageBase.isLoaded(level)){
        _imageBase.load(level);
        _imageOrientation.load(level);
        _imageId.load(level);
    }

    TexSynth::uploadImage_Color4(level, TS_LAYER_INPUT_COLOR, _imageBase);
    TexSynth::uploadImage_float(level, TS_LAYER_INPUT_ORIENTATION, _imageOrientation);
    TexSynth::uploadImage_int(level, TS_LAYER_INPUT_ID, _imageId);

    return true;
}


bool SynthesisFrame::loadAnimImages(int level, int time_step){

    __TIME_CODE_BLOCK("SynthesisFrame::loadAnimImages");

    DataAccess& data = DataAccess::instance();

    if (_anim_image_time_step != time_step) {
        clearAnimPyramids();
    }

    if(!_imageVelB.isInitialized())
    {
        data.getInputElement(IN_SURF_ID, _frame_number, _imageSurfaceId );
        if (time_step == 1) {
            data.getInputElement(IN_VEL_B, _frame_number, _imageVelB, DOWNSAMPLE_SCALE_WITH_LEVELS);
            data.getInputElement(IN_VEL_F,_frame_number, _imageVelF, DOWNSAMPLE_SCALE_WITH_LEVELS);
        } else {
            data.getRibbonB(_frame_number, time_step, _imageVelB);
            data.getRibbonF(_frame_number, time_step, _imageVelF);
        }
    }else if(!_imageVelB.isLoaded(level)){
        _imageSurfaceId.load(level);
        _imageVelB.load(level);
        _imageVelF.load(level);
    }

    TexSynth::uploadImage_float2(level, TS_LAYER_INPUT_SURF_ID, _imageSurfaceId);
    TexSynth::uploadImage_float2(level, TS_LAYER_INPUT_VEL_B, _imageVelB);
    TexSynth::uploadImage_float2(level, TS_LAYER_INPUT_VEL_F, _imageVelF);

    if(_do_keyframe_interpolation) {
        if (data.getBoolParameter("guidedInterpolation")) {
            _backwardGuide.load(level);
            _forwardGuide.load(level);
            TexSynth::uploadImage_Color4(level, TS_LAYER_GUIDE_B, _backwardGuide);
            TexSynth::uploadImage_Color4(level, TS_LAYER_GUIDE_F, _forwardGuide);
        }
        if (!_frameToKeyRibbonB.isInitialized()) {
            loadFrameToKeyRibbon(false);
            loadFrameToKeyRibbon(true);
        }
        _frameToKeyRibbonB.load(level);
        _frameToKeyRibbonF.load(level);
        TexSynth::uploadImage_Color4(level, TS_LAYER_FRAME_TO_KEY_RIBBON_B, _frameToKeyRibbonB);
        TexSynth::uploadImage_Color4(level, TS_LAYER_FRAME_TO_KEY_RIBBON_F, _frameToKeyRibbonF);
    }

    // Distance transform is used for advection culling calculation and so must be updated
    // with anim images.
    if(!_imageDistanceTransform.isInitialized())
    {
        data.getInputElement(IN_DIST_TRANS, _frame_number, _imageDistanceTransform);
    }
    else if(!_imageDistanceTransform.isLoaded(level)){
        _imageDistanceTransform.load(level);
    }
    TexSynth::uploadImage_float(level, TS_LAYER_INPUT_DIST_TRANS, _imageDistanceTransform);

    if(!_imageScale.isInitialized())
    {
	data.getInputElement(IN_SCALE, _frame_number, _imageScale);
    }
    else if(!_imageScale.isLoaded(level)){
	_imageScale.load(level);
    }
    TexSynth::uploadImage_float2(level, TS_LAYER_INPUT_SCALE, _imageScale);

    if (_anim_image_time_step != time_step) {
        //cacheAnimPyramids(-1, true);
        _anim_image_time_step = time_step;
    }

    return true;
}

bool SynthesisFrame::loadRibbonPair(int level, int previous_frame) {
    int current_frame = _frame_number;
    bool time_is_forwards = previous_frame < current_frame;
    int time_step = abs(previous_frame-current_frame);
    DataAccess& data = DataAccess::instance();

    ImagePyramid<float2> prev_pyr, cur_pyr;
    if (time_step == 1) {
        bool success = true;
        if (time_is_forwards) {
            success = success && data.getInputElement(IN_VEL_F, previous_frame, prev_pyr, DOWNSAMPLE_SCALE_WITH_LEVELS);
            success = success && data.getInputElement(IN_VEL_B, current_frame, cur_pyr, DOWNSAMPLE_SCALE_WITH_LEVELS);
        } else {
            success = success && data.getInputElement(IN_VEL_B, previous_frame, prev_pyr, DOWNSAMPLE_SCALE_WITH_LEVELS);
            success = success && data.getInputElement(IN_VEL_F, current_frame, cur_pyr, DOWNSAMPLE_SCALE_WITH_LEVELS);
        }
        if (!success) {
            return false;
        }
    } else {
        bool success = true;
        if (time_is_forwards) {
            success = success && data.getRibbonF(previous_frame, time_step, prev_pyr);
            success = success && data.getRibbonB(current_frame, time_step, cur_pyr);
        } else {
            success = success && data.getRibbonB(previous_frame, time_step, prev_pyr);
            success = success && data.getRibbonF(current_frame, time_step, cur_pyr);
        }
        if (!success) {
            return false;
        }
    }

    if (!pyramidSameSize(prev_pyr, cur_pyr)) {
        qWarning() << "Image size mismatch in loadRibbonPair frame " << current_frame;
        return false;
    }


    if (time_is_forwards) {
        TexSynth::uploadImage_float2(level, TS_LAYER_INPUT_VEL_B, cur_pyr);
        TexSynth::uploadImage_float2(level, TS_LAYER_INPUT_VEL_F_PREVIOUS, prev_pyr);
    } else {
        TexSynth::uploadImage_float2(level, TS_LAYER_INPUT_VEL_F, cur_pyr);
        TexSynth::uploadImage_float2(level, TS_LAYER_INPUT_VEL_B_PREVIOUS, prev_pyr);
    }

    return true;
}

void SynthesisFrame::cachePyramids(int level, bool overwrite)
{
    if (!_do_caching)
        return;

    __TIME_CODE_BLOCK("SynthesisFrame::cachePyramids");

    _imageBase.cache(level, overwrite);
    _imageOrientation.cache(level, overwrite);
    _imageRandom.cache(level, overwrite);
    _imageId.cache(level, overwrite);

    cacheAnimPyramids(level, overwrite);
}

void SynthesisFrame::cacheAnimPyramids(int level, bool overwrite)
{
    if (!_do_caching)
        return;

    __TIME_CODE_BLOCK("SynthesisFrame::cacheAnimPyramids");

    _imageDistanceTransform.cache(level, overwrite);
    _imageScale.cache(level, overwrite);
    _imageSurfaceId.cache(level, overwrite);
    _imageVelB.cache(level, overwrite);
    _imageVelF.cache(level, overwrite);

    _frameToKeyRibbonB.cache(level, overwrite);
    _frameToKeyRibbonF.cache(level, overwrite);

    if(_forwardGuide.isInitialized()){
        _forwardGuide.cache(level, overwrite);
        _backwardGuide.cache(level, overwrite);
    }
}

void SynthesisFrame::clearAnimPyramids()
{
    _imageDistanceTransform.clear();
    _imageScale.clear();
    _imageSurfaceId.clear();
    _imageVelB.clear();
    _imageVelF.clear();

    // Don't clear the frame to key ribbons or guide here,
    // because they are assumed to stay fixed over restarts and
    // levels.
}

bool SynthesisFrame::saveImages(int level, int pass, int version)
{
    if (!_working_buffers->saveTempImages(level, _frame_number, pass, version))
        return false;
    if (level > _last_saved_pass.size()) {
        _last_saved_pass.resize(level+1);
    }
    _last_saved_pass[level] = pass;
    return true;
}

bool SynthesisFrame::loadLastSavedOffsets(int level)
{
    return _working_buffers->loadOffsets(level, _frame_number, _last_saved_pass[level]);
}

bool SynthesisFrame::linkImagesToPass(int level, int pass)
{
    return _working_buffers->linkTempImagesToNextPass(level, _frame_number, _last_saved_pass[level], pass);
}
