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

#include <iostream>

#include <QtCore/QDir>
#include <QtCore/QDebug>
#include <QtCore/QFileInfo>
#include <QtCore/QFile>
#include <QtCore/QTime>

#include "synthesisProcessor.h"
#include "imageIO.h"
#include "dataAccess.h"
#include "progressCallback.h"

#include "stats.h"
#include "texSynth_kernel.h"

#include "workingBuffers.h"
#include "workSchedule.h"
#include "batchProcessor.h"
#include "cudaImagePyramid.h"
#include "cudaTexture.h"

using namespace std;

//----------------------------------------------------------------------------------------------------------------------------

//		Class Synthesis Processor:	

//----------------------------------------------------------------------------------------------------------------------------

typedef enum {
    STAGE_RENDER = 0x1,
    STAGE_KEYFRAME_PREPROCESS = 0x2,
    STAGE_INVALID = 0x0
} SynthesisStage;


#define WAIT_AND_RETURN_IF_TERMINATED(msg) _thread_state.waitForContinue(msg); if (_thread_state._synthesisState == STATE_TERMINATING) return false;

//----------------------------------------------------------------------------------------------------------------------------

bool SynthesisThreadData::waitForContinue(const char* msg)
{
    if (_synthesisState == STATE_TERMINATING)
        return false;
    if (_synthesisState == STATE_RUNNING_NO_WAITING)
        return true;

    _synthesisState = STATE_WAITING_FOR_CONTINUE;
    _waitMessage = QString(msg);
    _waiter.wait(&_mutex);

    if (_synthesisState == STATE_CONTINUING) {
        _synthesisState = STATE_WORKING;
        return true;
    } else if (_synthesisState == STATE_TERMINATING) {
        return false;
    } else {
        assert(0);
    }
    return false;
}

//----------------------------------------------------------------------------------------------------------------------------

SynthesisProcessor::SynthesisProcessor(DataAccess* data)
    : _iterations(3),
      _iterationsFirstFrame(3),
      _levels(1),
      _finalLevel(0),
      _renderPreviewStill(false),
      _keyFrameInterpolation(false),
      _firstFrame(1),
      _lastFrame(1),
      _fixupPasses(0),
      _firstPassDirection(FORWARD),
      _savePreprocessToOutput(true),
      _preprocessed(false),
      _data(data),
      _work_schedule(NULL)
{
    _cudaWorkingBuffers = new WorkingBuffers();
    _workerThread = new Worker(this);
    _thread_state._synthesisState = STATE_READY;
}

SynthesisProcessor::~SynthesisProcessor()
{
    _workerThread->terminate();
    _workerThread->wait();
    delete _workerThread;
    delete _cudaWorkingBuffers;
}

void SynthesisProcessor::copyOutputToFinal(int final_pass)
{
    QDir out_dir = QFileInfo(ImageIO::parsePath(_data->getOutPath(OUT_OUTPUT_FINAL).arg(""),0)).dir();
    for (int frame = _firstFrame; frame <= _lastFrame; frame++) {
        QString temp_path = ImageIO::temporaryOutputPath(_data->getOutPath(OUT_OUTPUT),0,frame,final_pass,0);
        QString final_path = ImageIO::parsePath(_data->getOutPath(OUT_OUTPUT_FINAL).arg(""),frame);

        QFile temp_file(temp_path);
        if (!temp_file.exists()) {
            qWarning("Could not find %s", qPrintable(temp_path));
            continue;
        }

        out_dir.remove(final_path);
        temp_file.copy(final_path);
    }

}

// Has to be called after si_updateParameters to ensure correct frame numbers
void SynthesisProcessor::initializeFrameStorage(){

    int nbFrames = _lastFrame - _firstFrame + 1;
    _storageFirstFrame = _firstFrame;

    if (_keyFrameInterpolation) {
        _style.updateKeyFrameSubrange(_firstFrame, _lastFrame);
        if(_style.keyFrameSubrange().empty()) {
            qDebug("Empty key frame subrange in initialzeFrameStorage");
            return;
        }

        _storageFirstFrame = min(_firstFrame,_style.keyFrameSubrange().first());
        nbFrames = max(_lastFrame,_style.keyFrameSubrange().last()) - _storageFirstFrame + 1;
    }

    _frames.clear();
    _frames.resize(nbFrames);
    for (int i = 0; i < nbFrames; i++) {
        _frames[i].initialize(_storageFirstFrame + i, &_style, _cudaWorkingBuffers, &_thread_state);
    }
}


// Function to grab the parameters:
void SynthesisProcessor::updateParameters(bool for_offline_synthesis, bool load_default = true)
{
    qDebug()<<"Synthesis Processor: Updating parameters";

    if (!_data) {
        qCritical() << "SynthesisProcessor: Could not update parameters (_data == NULL)! ";
        return;
    }

    TsParameters tsParams;
    if (load_default)
        tsParams = _data->getTsDefaultParams();
    else
        tsParams = _cudaWorkingBuffers->params();

    tsParams.residualWindowSize = _data->getIntParameter("residualWindowSize");
    tsParams.coherenceWeight = _data->getFloatParameter("coherenceWeight");
    tsParams.distTransWeight = _data->getFloatParameter("distanceTransformWeight");
    tsParams.inputAnalogyWeight = _data->getFloatParameter("inputAnalogyWeight");
    tsParams.canvasOutputWeight = _data->getFloatParameter("canvasOutputWeight");
    tsParams.scatterSamplesCount = _data->getIntParameter("scatterSamplesCount");
    tsParams.timeDerivativeInputWeight = _data->getFloatParameter("timeDerivativeInputWeight");
    tsParams.timeDerivativeOutputWeight = _data->getFloatParameter("timeDerivativeOutputWeight");
    tsParams.temporalCoherenceWeight = _data->getFloatParameter("temporalCoherenceWeight");
    tsParams.hysteresisWeight = _data->getFloatParameter("hysteresisWeight");
    tsParams.maxDistortion = _data->getFloatParameter("maxDistortion");
    tsParams.advectionJitter = _data->getFloatParameter("advectionJitter") * 0.01;

    assert(_data->getNumStyles() < NUM_MAX_STYLES);

    QVector<float> slopes = _data->getOffsetsHistogramSlopes();
    QVector<float> thresholds = _data->getOffsetsHistogramThresholds();
    for(int i=0; i < _data->getNumStyles(); ++i){
        tsParams.offsetsHistogramSlope[i] = slopes[i];
        tsParams.offsetsHistogramThreshold[i] = thresholds[i];
    }

    tsParams.numStyles = _data->getNumStyles();
    tsParams.alphaWeight = _data->getFloatParameter("alphaChannelWeight");
    tsParams.transparencyOk = _data->getBoolParameter("transparencyAllowed");
    tsParams.colorspace = _data->getIntParameter("colorspace");
    tsParams.orientationEnabled = _data->getBoolParameter("orientationEnabled");

    tsParams.styleObjIdWeight = _data->getFloatParameter("styleObjectIdWeight");

    tsParams.lumShiftEnable = _data->getBoolParameter("lumShiftEnabled");
    tsParams.lumShiftWeight = _data->getFloatParameter("lumShiftWeight");

    tsParams.levels = _data->getIntParameter("levels");

    tsParams.residualWindowScaleU = _data->getFloatParameter("residualWindowScaleU");
    tsParams.residualWindowScaleV = _data->getFloatParameter("residualWindowScaleV");

    tsParams.distTransCull = 50;

    tsParams.numFrames = _lastFrame - _firstFrame;
    tsParams.useGuide = _data->getBoolParameter("guidedInterpolation");

    // Grab levels:
    _levels = _data->getIntParameter("levels");
    _finalLevel = _data->getIntParameter("maxLevel");
    _fixupPasses = _data->getIntParameter("fixupPasses");

    // Grab animation data and flags:
    _renderPreviewStill = !for_offline_synthesis && _data->getRealtimeSynthesisMode() == 0;
    if (_renderPreviewStill) {
        _firstFrame = _data->getCurrentPreviewFrame();
        _lastFrame = _data->getCurrentPreviewFrame();
        _storageFirstFrame = _firstFrame;
        _iterationsFirstFrame = _iterations = _data->getIntParameter("iterations");
        _keyFrameInterpolation = false;
    } else {
        _firstFrame = _data->firstFrame();
        _lastFrame = _data->lastFrame();
        _storageFirstFrame = _firstFrame;
        _iterations = _data->getIntParameter("iterations");
        _keyFrameInterpolation = _data->getBoolParameter("keyFrameInterpolation");
        _style.updateKeyFrameSubrange(_firstFrame, _lastFrame);
        if(_style.keyFrameSubrange().size() > 0){
            tsParams.firstKeyIndex = _style.firstKeyFrameFullIndex();
            tsParams.lastKeyIndex = _style.lastKeyFrameFullIndex();
            _storageFirstFrame =  min(_firstFrame,_style.keyFrameSubrange().first());
        }else{
            tsParams.lastKeyIndex = -1;
            tsParams.lastKeyIndex = -1;
        }
    }
    SynthesisFrame::setSynthesisRange(_firstFrame, _lastFrame, _levels,
                                      _keyFrameInterpolation, _data->getBoolParameter("run_local"));

    tsParams.interpolateKeyFrame = _keyFrameInterpolation;

    _cudaWorkingBuffers->setParams(tsParams);
    TexSynth::uploadParameters(tsParams);

    _firstPassDirection = _data->getFirstPassDirection();
}


// Function to save current synthesized image:
void SynthesisProcessor::si_saveImage(const QString& filePath)
{
    _cudaWorkingBuffers->saveFinalOutput(filePath);
}

bool SynthesisProcessor::copyInputArray(InputElements element, cudaArray* array)
{
    int curWidth = _cudaWorkingBuffers->currentImageWidth();
    int curHeight = _cudaWorkingBuffers->currentImageHeight();
    switch(element){
    case IN_INPUT:
        TexSynth::copyInput(TS_LAYER_INPUT_COLOR, array, curWidth, curHeight);
        break;
    case IN_ORIENTATION:
        TexSynth::copyInput(TS_LAYER_INPUT_ORIENTATION, array, curWidth, curHeight);
        break;
    case IN_DIST_TRANS:
        TexSynth::copyInput(TS_LAYER_INPUT_DIST_TRANS, array, curWidth, curHeight);
        break;
    case IN_SCALE:
	TexSynth::copyInput(TS_LAYER_INPUT_SCALE, array, curWidth, curHeight);
	break;
    default:
        return false;
    }
    return true;
}

bool SynthesisProcessor::copyStyleArray(StyleElements element, int numStyle, cudaArray* array)
{
    int curWidth = _cudaWorkingBuffers->currentStyleWidth(numStyle);
    int curHeight = _cudaWorkingBuffers->currentStyleHeight(numStyle);
    switch(element){
    case STYLE_INPUT:
        TexSynth::copyInput(TS_LAYER_EXEMPLAR_BASE, array, curWidth, curHeight, numStyle);
        break;
    case STYLE_OUTPUT:
        TexSynth::copyInput(TS_LAYER_EXEMPLAR_OUTPUT, array, curWidth, curHeight, numStyle);
        break;
    case STYLE_ORIENTATION:
        TexSynth::copyInput(TS_LAYER_EXEMPLAR_ORIENTATION, array, curWidth, curHeight, numStyle);
        break;
    case STYLE_DIST_TRANS:
        TexSynth::copyInput(TS_LAYER_EXEMPLAR_DIST_TRANS, array, curWidth, curHeight, numStyle);
        break;
    default:
        return false;
    }
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------
// Functions for controlling the stylization (texture synthesis)
// in a step-wise manner (doing one processing step at a time):

bool SynthesisProcessor::si_realtime_advanceSynthesis(bool take_snapshots)
{
    if (!take_snapshots) {
        // If running full speed, forgot about the history.
        _cudaWorkingBuffers->clearHistory();
    } else {
        // If we are paging through the history snapshots,
        // just advance.
        if (_cudaWorkingBuffers->stepHistoryForward())
            return true;
    }

    // Special case to take snapshot immediately after pausing
    // or restarting
    if (take_snapshots && _cudaWorkingBuffers->historySize() == 0) {
        _cudaWorkingBuffers->takeHistorySnapshot(_thread_state._waitMessage);
    }

    // Otherwise, prompt a new step of synthesis.
    _thread_state._synthesisState = STATE_CONTINUING;
    _thread_state._waiter.wakeAll();
    // Wait for the worker to perform a step.
    // It will change the synthesis state first to STATE_WORKING,
    // then either WAITING or TERMINATING when it's done.
    // Should really use a semaphore or something here.
    do {
        Worker::sleep(1); // static member => the main thread go to sleep not the worker thread
    } while (_thread_state._synthesisState == STATE_CONTINUING);
    do {
        Worker::sleep(1);
    } while (_thread_state._synthesisState == STATE_WORKING);

    //_cudaWorkingBuffers->setCurrentLevel(_currentLevel);
    if (take_snapshots) {
        _cudaWorkingBuffers->takeHistorySnapshot(_thread_state._waitMessage);
    }

    if (_thread_state._synthesisState == STATE_WORKING ||
            _thread_state._synthesisState == STATE_WAITING_FOR_CONTINUE){
        return true;
    } else {
        return false;
    }
}

bool SynthesisProcessor::si_realtime_backtrackSynthesis()
{
    return _cudaWorkingBuffers->stepHistoryBack();
}

//----------------------------------------------------------------------------------------------------------------------------
// Functions to start and stop a synthesis: (synthesis setup/cleanup)

void SynthesisProcessor::Worker::shutDown()
{
    if (isRunning()) {
        // Notify the worker that it should exit.
        _parent->_thread_state._synthesisState = STATE_TERMINATING;
        _parent->_thread_state._waiter.wakeAll();
        wait();
    }
}

void SynthesisProcessor::Worker::runStages(int stages)
{
    _stages = stages;
    QThread::start();
}

void SynthesisProcessor::Worker::run()
{
    _parent->_thread_state._mutex.lock();

    bool abort = false;
    bool run_batch = !(DataAccess::instance().getStringParameter("schedule_file").isEmpty());

    if (run_batch) {
        if (!_parent->runBatch())
            abort = true;
    } else {
        if (_stages & STAGE_KEYFRAME_PREPROCESS) {
            if (!_parent->keyFramesPreprocess())
                abort = true;
            else
                _parent->_preprocessed = true;
        }
        if (!abort) {
            if (!_parent->renderAnimation())
                abort = true;
        }
    }
    if (!abort) {
        _parent->_thread_state._synthesisState = STATE_FINISHED;
    } else {
        _parent->_thread_state._synthesisState = STATE_ABORTED;
    }

    _parent->_thread_state._mutex.unlock();
}

void SynthesisProcessor::si_realtime_initSynthesis(bool cleanStyle, bool cleanImage)
{
    _workerThread->shutDown();

    updateParameters(false);

    _progress = 0;
    _print_progress = false;

    if(cleanStyle){
        _style.clear();
    }
    _style.load(_data->getCurrentPreviewLevel());

    _cudaWorkingBuffers->initialize();

    int level = _data->getCurrentPreviewLevel();
    if(_renderPreviewStill || !_keyFrameInterpolation) {
        if(cleanImage)
            initializeFrameStorage();
        SynthesisFrame* frame = framePtr(_data->getCurrentPreviewFrame());
        frame->loadImages(level);
        frame->loadAnimImages(level,1);
        frame->createRandomPyramid(level);
    }
    _cudaWorkingBuffers->setCurrentLevel(level);
}

void SynthesisProcessor::si_realtime_startSynthesis()
{
    _workerThread->shutDown();

    updateParameters(false);

    int stages = STAGE_RENDER;
    if (_keyFrameInterpolation) {
        stages |= STAGE_KEYFRAME_PREPROCESS;
    }

    _thread_state._synthesisState = STATE_WORKING;
    _workerThread->runStages(stages);
}

void SynthesisProcessor::si_cancelSynthesis()
{
    _workerThread->shutDown();
    _cudaWorkingBuffers->clearHistory();

    updateParameters(false);

    _thread_state._synthesisState = STATE_READY;
}

void SynthesisProcessor::si_cleanupSynthesis()
{
    _workerThread->shutDown();

    _cudaWorkingBuffers->clear();
    CudaImagePyramidHost::clearAllInstances();
    CudaTextureHost::clearAllInstances();
}

void SynthesisProcessor::si_parameterChanged()
{
    if (_cudaWorkingBuffers->isInitialized() && _workerThread->isRunning()) {
        updateParameters(false, false);
        TexSynth::updateResidualCache(_cudaWorkingBuffers, _cudaWorkingBuffers->currentLevel());
    }
}

void SynthesisProcessor::si_realtime_restartSynthesis()
{
    _workerThread->shutDown();

    _cudaWorkingBuffers->clearHistory();

    updateParameters(false);

    int stages = STAGE_RENDER;
    if (_keyFrameInterpolation && !_preprocessed) {
        stages |= STAGE_KEYFRAME_PREPROCESS;
    }

    _thread_state._synthesisState = STATE_WORKING;
    _workerThread->runStages(stages);
}

void resetRibbonField(QVector<RibbonP>& ribbon, int layer, int width, int height)
{
    ribbon.resize(width*height);
    for (int i = 0; i < width*height; i++) {
        RibbonP p;
        p.layer = layer;
        ribbon[i] = p;
    }
}

SynthesisFrame* SynthesisProcessor::framePtr(int frame_number)
{
    int index = frame_number - _storageFirstFrame;
    if (index < 0 || index >= _frames.size()) {
        return NULL;
    }
    return &_frames[index];
}

bool SynthesisProcessor::accumulateRibbon(bool timeIsForwards, int width, int height)
{
    QVector<Color4> advectedColors;
    QVector< QVector<RibbonP> > accumulatedRibbons;

    int level = 0; // _maxLevel in the future;
    int firstKey = _style.keyFrameSubrange().first();
    int lastKey = _style.keyFrameSubrange().last();

    _storageFirstFrame = min(_firstFrame,firstKey);
    int storageLastFrame = max(_lastFrame, lastKey);
    int nbFrames = max(_lastFrame,lastKey) - _storageFirstFrame + 1;
    accumulatedRibbons.resize(nbFrames);

    QVector<int> accumulateFrames, blankFrames;
    if (timeIsForwards) {
        for (int frame=_firstFrame; frame<firstKey; frame++){
            blankFrames.push_back(frame);
        }
        for(int frame=firstKey; frame<=storageLastFrame; frame++){
            accumulateFrames.push_back(frame);
        }
    } else {
        for (int frame=lastKey+1; frame<=_lastFrame; frame++){
            blankFrames.push_back(frame);
        }
        for(int frame=lastKey; frame>=_storageFirstFrame; frame--){
            accumulateFrames.push_back(frame);
        }
    }

    _data->setCurrentPreviewLevel(0);

    int first_key_index = accumulateFrames.first()-_storageFirstFrame;
    resetRibbonField(accumulatedRibbons[first_key_index], -1, width, height);
    advectedColors.resize(width*height);
    _cudaWorkingBuffers->ribbonField.base().copyFromHost(accumulatedRibbons[first_key_index],level);

    SynthesisFrame* framep;
    foreach(int frame, blankFrames) {
        framep = framePtr(frame);
        framep->initializeGuide(advectedColors, width, height, timeIsForwards, _savePreprocessToOutput);
        framep->initializeFrameToKeyRibbon(accumulatedRibbons[first_key_index], width, height, timeIsForwards, _savePreprocessToOutput);
        framep->cachePyramids(-1,true);
    }

    foreach(int frame, accumulateFrames) {
        _data->setCurrentPreviewFrame(frame);
        framep = framePtr(frame);
        int time_step = (timeIsForwards) ? -1 : 1;
        QString status = QString("Advect from frame %1 to frame %2... ").arg(frame+time_step).arg(frame);
        cout << qPrintable(status) << flush;

        framep->loadImages(0);

        int idx = frame-_storageFirstFrame;
        while ((frame + time_step >= firstKey && timeIsForwards) ||
               (frame + time_step <= lastKey && !timeIsForwards)) {
            if (!framep->loadRibbonPair(0,frame+time_step))
                break;

            //cout << time_step << " " << std::flush;

            int prev_idx = idx + time_step;
            _cudaWorkingBuffers->ribbonField.base().copyFromHost(accumulatedRibbons[prev_idx], level);
            bool overwrite = abs(time_step) == 1;
            TexSynth::accumulateRibbonField(_cudaWorkingBuffers,level,timeIsForwards,time_step,overwrite);

            time_step *= 2;

            //WAIT_AND_RETURN_IF_TERMINATED(status.toAscii());
        }

        _cudaWorkingBuffers->advectedF.output().copyToHost(advectedColors,level);
        _cudaWorkingBuffers->ribbonField.base().copyToHost(accumulatedRibbons[idx], level);

        int key_index = _style.keyFrameSubrangeIndex(frame);
        if(key_index >= 0){
            // The key-to-key ribbon at a key frame does *not* include the current key,
            // so save it before we reset the accumulated buffer.
            _style.initializeKeyToKeyRibbon(accumulatedRibbons[idx], frame, width, height, timeIsForwards, _savePreprocessToOutput);

            _data->getStyle(STYLE_OUTPUT, _style.keyFrameFullIndex(frame), advectedColors);
            _cudaWorkingBuffers->canvasOutputCache.copyFromHost(advectedColors,level);
            _cudaWorkingBuffers->advectedF.output().copyFromHost(advectedColors, level); // Just for visualization.
            resetRibbonField(accumulatedRibbons[idx], key_index, width, height);
            _cudaWorkingBuffers->ribbonField.base().copyFromHost(accumulatedRibbons[idx],level);
        }

        framep->initializeGuide(advectedColors, width, height, timeIsForwards, _savePreprocessToOutput);
        framep->initializeFrameToKeyRibbon(accumulatedRibbons[idx], width, height, timeIsForwards, _savePreprocessToOutput);
        framep->cachePyramids(-1,true);

        std::cout << "done." << std::endl;

        WAIT_AND_RETURN_IF_TERMINATED(status.toStdString().c_str());
    }

    return true;
}

bool SynthesisProcessor::keyFramesPreprocess()
{
    initializeFrameStorage();
    if(_style.keyFrameSubrange().empty())
        return true;

    _style.load(0);

    // Stupid... have to load an image before we know how big the images are.
    SynthesisFrame* framep = framePtr(_style.keyFrameSubrange().first());
    framep->loadImages(0);
    int width = framep->imageBase().width();
    int height = framep->imageBase().height();

    if (_data->getBoolParameter("use_cached_preprocess")) {
        for (int frame = _firstFrame; frame <= _lastFrame; frame++) {
            framep = framePtr(frame);
            QString status = QString("Reading preprocessed inputs for frame %1... ").arg(frame);
            cout << qPrintable(status) << std::flush;
            framep->loadGuide(true);
            framep->loadGuide(false);
            framep->loadFrameToKeyRibbon(true);
            framep->loadFrameToKeyRibbon(false);
            if (_style.keyFrameSubrange().contains(frame)) {
                _style.loadKeyToKeyRibbon(frame, true);
                _style.loadKeyToKeyRibbon(frame, false);
            }
            framep->cachePyramids(-1,true);
            std::cout << "done." << std::endl;
        }
    } else {
        if (!accumulateRibbon(true, width, height))
            return false;
        if (!accumulateRibbon(false, width, height))
            return false;
    }

    // Synthesize the animation with key-frame interpolation
    _keyFrameInterpolation = true;
    _cudaWorkingBuffers->params().interpolateKeyFrame = true;
    _cudaWorkingBuffers->params().numFrames = _frames.size()-1;
    _cudaWorkingBuffers->params().useGuide = _data->getBoolParameter("guidedInterpolation");
    _cudaWorkingBuffers->params().firstKeyIndex = _style.firstKeyFrameFullIndex();
    _cudaWorkingBuffers->params().lastKeyIndex = _style.lastKeyFrameFullIndex();
    return true;
}

void SynthesisProcessor::si_offline_initSynthesis()
{
    updateParameters(true);
    initializeFrameStorage();

    _cudaWorkingBuffers->initialize();
    _cudaWorkingBuffers->clearHistory();

    _thread_state._synthesisState = STATE_RUNNING_NO_WAITING;
}

void SynthesisProcessor::si_offline_startSynthesis(ProgressCallback* progress)
{
    _progress = 0;
    _print_progress = false;

    si_offline_initSynthesis();

    if(_keyFrameInterpolation)
        _preprocessed = keyFramesPreprocess();

    _progress = progress;
    _print_progress = true;

    renderAnimation();
}

bool SynthesisProcessor::renderConsecutivePass(int level, int pass,
                                               int source_pass, int time_step)
{
    QList<int> frames;
    bool time_is_forwards = (pass % 2 == _firstPassDirection);

    if (time_is_forwards) {
        for (int i = _firstFrame; i <= _lastFrame; i += time_step) {
            frames.push_back(i);
        }
    } else {
        for (int i = _lastFrame; i >= _firstFrame; i -= time_step) {
            frames.push_back(i);
        }
    }

    if (pass > 0) {
        // If on a fixup pass, skip synthesis of the first frame and
        // just copy the last frame synthesized (still in memory) as the
        // starting frame.
        _cudaWorkingBuffers->saveTempImages(level, frames[0], pass, 0);
        frames.pop_front();
    }

    for(int i = 0; i < frames.size(); i++){
        int frame_number = frames[i];
        updateProgress(level, pass, i);

        std::cout << "Level " << level << ", pass "<< pass << ", frame " << frame_number << "..." << std::flush;

        bool is_first_frame = (i == 0) && (pass == 0);

        if(!is_first_frame){
            // Move stuff into backup buffers for advection step (later):
            TexSynth::flipAnimBuffers();
        }
        SynthesisFrame* frame = framePtr(frame_number);

        frame->loadImages(level);
        frame->loadAnimImages(level, time_step);
        _data->setCurrentPreviewFrame(frame_number);

        if (!frame->synthesizeConsecutive(level, pass, time_step,
                                    is_first_frame, time_is_forwards, source_pass, pass==0))
            return false;

        std::cout << "done." << std::endl;
    }

    return true;
}

bool SynthesisProcessor::runBatch()
{
    DataAccess& data = DataAccess::instance();

    WorkSchedule schedule;
    if (!schedule.load(data.getStringParameter("schedule_file")))
        return false;

    QString tasks = data.getStringParameter("tasks");
    BatchProcessor bp(this);

    return bp.runSchedule(schedule, tasks);
}

bool SynthesisProcessor::renderAnimation()
{
    QTime timer;
    timer.start();

    qDebug() << "Starting animation render from " << _firstFrame << " to " << _lastFrame
             << " with " << _levels << " levels and " << _fixupPasses << " fixup passes";
    qDebug() << "Writing temporary output to " << _data->getTemporaryDir() << ", final output to "
             << _data->getOutDir();

    TexSynth::setup(_cudaWorkingBuffers->params());

    bool success = false;
    switch(_data->getSynthesisScheme()){
    case S_FMBM:
    case S_FMBM_RM:
        success = renderAnimationFMBM();
        break;
    case S_FBMO:
        success = renderAnimationFBMO();
        break;
    case S_TCTF:
        success = renderAnimationTCTF();
        break;
    case S_IND:
        success = renderAnimationIND();
        break;
    }

    if (success) {
        qDebug()<<"Animation synthesis finished ("<< timer.elapsed() / 1000 <<" s)";
    } else {
        qDebug()<<"Animation synthesis aborted";
    }

    return success;
}

bool SynthesisProcessor::renderAnimationIND()
{
    int num_total_passes = _fixupPasses + 1;
    for (int i = _firstFrame; i <= _lastFrame; i ++) {
        // synthesize each frame independently
        for(int level = _levels-1; level>=_finalLevel; level--) {
            _data->setCurrentPreviewLevel(level);
            _style.load(level);

            _cudaWorkingBuffers->resetAdvectedBuffers();
            _cudaWorkingBuffers->setCurrentLevel(level);
            for(int pass = 0; pass < num_total_passes; pass++) {
                _data->setCurrentPreviewPass(pass);

                int frame_number = i;

                std::cout << "Level " << level << ", pass "<< pass << ", frame " << frame_number << "..." << std::flush;

                SynthesisFrame* frame = framePtr(frame_number);

                frame->loadImages(level);
                frame->loadAnimImages(level, 1);
                _data->setCurrentPreviewFrame(frame_number);

                if (!frame->synthesizeConsecutive(level, pass, 1,
                                            true, true, pass, pass==0))
                    return false;

                std::cout << "done." << std::endl;
            }
        }
    }
    int final_pass = (_finalLevel == 0) ? num_total_passes-1 : 0;

    copyOutputToFinal(final_pass);

    return true;
}

bool SynthesisProcessor::renderAnimationFMBM()
{
    int num_total_passes = _fixupPasses + 1;
    for(int level = _levels-1; level>=_finalLevel; level--) {
        _data->setCurrentPreviewLevel(level);
        _style.load(level);

        _cudaWorkingBuffers->resetAdvectedBuffers();
        _cudaWorkingBuffers->setCurrentLevel(level);

        for(int pass = 0; pass < num_total_passes; pass++) {
            _data->setCurrentPreviewPass(pass);
            if (!renderConsecutivePass(level, pass, num_total_passes-1, 1))
                return false;
        }
    }

    for (int level = _finalLevel; level > 0; level--) {
        int source_pass = (level == _finalLevel) ? num_total_passes-1 : 0;

        _cudaWorkingBuffers->setCurrentLevel(level-1);
        _style.load(level-1);
        if (!renderSpatialUpsample(level, source_pass))
            return false;
    }
    _data->setCurrentPreviewLevel(1);

    int final_pass = (_finalLevel == 0) ? num_total_passes-1 : 0;

    copyOutputToFinal(final_pass);

    return true;
}

bool SynthesisProcessor::renderInterpolatedFixupPass(int level, int pass, int time_step, int which_frames)
{
    QVector<int> framesIndices;
    if (which_frames == REFINE_ODD) {
        for(int i = _firstFrame + time_step; i <= _lastFrame; i += 2*time_step) {
            framesIndices.push_back(i);
        }
    } else {
        for(int i = _firstFrame; i <= _lastFrame; i += time_step) {
            framesIndices.push_back(i);
        }
    }

    for(int i=0; i<framesIndices.size(); i++){
        int frame = framesIndices[i];

        if (_work_schedule) {
            _work_schedule->addSynthesisInterpolated(level, pass, frame, time_step);
        } else {
            _data->setCurrentPreviewFrame(frame);
            std::cout << "Level " << level << ", pass "<< pass << ", frame " << frame << "..." << std::flush;

            SynthesisFrame* this_frame = framePtr(frame);
            SynthesisFrame* prior_frame = framePtr(frame - time_step);
            SynthesisFrame* next_frame = framePtr(frame + time_step);

            this_frame->createRandomPyramid(level);
            if (!this_frame->synthesizeInterpolated(level, pass, pass-1, time_step, prior_frame, next_frame))
                return false;

            std::cout << " done." << std::endl;
        }
    }

    // If fixup only odd, copy the even steps forward to keep indexing consistent.
    if (which_frames == REFINE_ODD) {
        for (int frame = _firstFrame; frame <= _lastFrame; frame += 2*time_step) {
            std::cout << "Link even frame, Level " << level << ", pass "<< pass << ", frame " << frame << "..." << std::flush;
            _cudaWorkingBuffers->linkTempImagesToNextPass(level, frame, pass-1, pass);
            std::cout << " done." << std::endl;
        }
    }

    return true;
}

bool SynthesisProcessor::renderAnimationFBMO()
{
    for(int level = _levels-1; level>=_finalLevel; level--){
        _data->setCurrentPreviewLevel(level);
        _style.load(level);

        // Passes 0 & 1: Independant forward and backward synthesis
        for(int pass = 0; pass < 2; pass++){
            _data->setCurrentPreviewPass(pass);

            _cudaWorkingBuffers->resetAdvectedBuffers();

            bool time_is_forwards = (pass % 2 == _firstPassDirection);

            int sFrame, eFrame;
            int fStep = -1;

            if(time_is_forwards){
                sFrame = _firstFrame;
                eFrame = _lastFrame+1;
                fStep = 1;
            }else{
                sFrame = _lastFrame;
                eFrame = _firstFrame-1;
            }

            for(int frame = sFrame; frame != eFrame; frame+=fStep){
                _data->setCurrentPreviewFrame(frame);
                // Inform of next computation:
                updateProgress(level, pass, frame);

                bool is_first_frame = (time_is_forwards && frame == _firstFrame) || (!time_is_forwards && frame == _lastFrame);

                if(!is_first_frame || pass > 0){
                    // Move stuff into backup buffers for advection step (later):
                    TexSynth::flipAnimBuffers();
                }

                framePtr(frame)->loadImages(level);
                framePtr(frame)->loadAnimImages(level, 1);

                std::cout << "Level " << level << ", pass "<< pass << ", frame " << frame << "...";

                _cudaWorkingBuffers->setCurrentLevel(level);

                framePtr(frame)->synthesizeConsecutive(level, pass, 1, is_first_frame, time_is_forwards, _fixupPasses + 2, true);

                std::cout << " done.\n" << std::flush;

            } // frame
        } // pass

        _data->setCurrentPreviewPass(2);
        // Pass 2 : merge backward and forward passes
        for(int frame = _firstFrame; frame != _lastFrame+1; frame++){
            _data->setCurrentPreviewFrame(frame);

            std::cout << "Level " << level << ", pass 2" << ", frame " << frame << "...";

            _cudaWorkingBuffers->loadOffsets(level, frame, 0);
            _cudaWorkingBuffers->loadResidual(level, frame, 0);
            _cudaWorkingBuffers->offsets.base().copyTo(_cudaWorkingBuffers->advectedF.offsets());
            _cudaWorkingBuffers->residualCache.base().copyTo(_cudaWorkingBuffers->advectedF.residual());

            TexSynth::syncBuffers(_cudaWorkingBuffers, level, TexSynth::DONT_COPY_OFFSETS | TexSynth::DONT_UPDATE_RESIDUAL);
            // WAIT_AND_RETURN_IF_TERMINATED("load forward");

            _cudaWorkingBuffers->loadOffsets(level, frame, 1);
            _cudaWorkingBuffers->loadResidual(level, frame, 1);

            TexSynth::syncBuffers(_cudaWorkingBuffers, level, TexSynth::DONT_COPY_OFFSETS | TexSynth::DONT_UPDATE_RESIDUAL);
            // WAIT_AND_RETURN_IF_TERMINATED("load backward");

            framePtr(frame)->mergeRandomly(level, true, true);

            WAIT_AND_RETURN_IF_TERMINATED("after merge");

            _cudaWorkingBuffers->saveTempImages(level, frame, 2, 0);

            std::cout << " done.\n" << std::flush;
        }

        _cudaWorkingBuffers->params().direction = BIDIRECTIONAL;
        TexSynth::uploadParameters(_cudaWorkingBuffers->params());

        // Fixup passes: random frame optimization
        for(int pass = 3; pass < _fixupPasses+3; pass++){
            _data->setCurrentPreviewPass(pass);
            if (!renderInterpolatedFixupPass(level, pass, 1, REFINE_EVEN_AND_ODD))
                return false;
        }
    } // level

    copyOutputToFinal(_fixupPasses+2);

    return true;
}


bool SynthesisProcessor::renderSpatialUpsample(int coarse_level, int source_pass)
{
    int fine_level = coarse_level - 1;

    for (int frame_number = _firstFrame; frame_number <= _lastFrame; frame_number++) {

        if (_work_schedule) {
            _work_schedule->addSpatialUpsample(fine_level, source_pass, frame_number);
        } else {
            std::cout << "Spatial upsample, level " << coarse_level << " to " << fine_level << ", frame " << frame_number << "...";

            bool ret = framePtr(frame_number)->spatialUpsample(coarse_level, fine_level, source_pass);
            if (!ret)
                return false;

            std::cout << " done.\n" << std::flush;
        }
    }

    return true;
}

bool SynthesisProcessor::renderInitializationPass(int level, int time_step, int source_pass)
{
    QList<int> frames;
    for (int i = _firstFrame; i <= _lastFrame; i += time_step) {
        frames.push_back(i);
    }

    int init_mode = SynthesisFrame::INIT_SPATIAL_UPSAMPLE;
    if (level == _levels - 1) {
        init_mode = SynthesisFrame::INIT_TEMPORAL_UPSAMPLE;
    }

    for (int i = 0; i < frames.size(); i++) {
        int frame_number = frames[i];

        if (_work_schedule) {
            _work_schedule->addInitialization(level, source_pass, time_step, frame_number, init_mode);
        } else {
            std::cout << "Initialization, time step " << time_step << ", level " << level << ", frame " << frame_number << "...";

            bool ret = framePtr(frame_number)->refine(level, 0, source_pass, time_step, 0, 0, init_mode);
            if (!ret)
                return false;

            std::cout << " done.\n" << std::flush;
        }
    }

    return true;
}

bool SynthesisProcessor::renderRefinePass(int level, int last_coarse_pass, int this_pass, int time_step, SynthesisOperation op)
{
    int coarse_step = time_step * 2;
    //int source_pass = this_pass - 1;

    QVector<int> frames;
    if (op == REFINE_ODD) {
        for (int i = _firstFrame + time_step; i <= _lastFrame; i += coarse_step) {
            frames.push_back(i);
        }
    } else if (op == REFINE_EVEN) {
        for (int i = _firstFrame; i <= _lastFrame; i += coarse_step) {
            frames.push_back(i);
        }
    } else {
        for (int i = _firstFrame; i <= _lastFrame; i += time_step) {
            frames.push_back(i);
        }
    }

    int init_mode = SynthesisFrame::INIT_PREVIOUS_PASS;
    if (op == REFINE_ODD) {
        init_mode = SynthesisFrame::INIT_SPATIAL_UPSAMPLE;
        if (level == _levels-1) {
            init_mode = SynthesisFrame::INIT_TEMPORAL_UPSAMPLE;
        }
    } else if (op == REFINE_EVEN_AND_ODD && (last_coarse_pass == this_pass-1)) {
        init_mode = SynthesisFrame::INIT_SPATIAL_UPSAMPLE;
    }

    for (int i = 0; i < frames.size(); i++) {
        int frame_number = frames[i];

        if (_work_schedule) {
            _work_schedule->addRefine(level, last_coarse_pass, this_pass, time_step, frame_number, init_mode);
        } else {
            _data->setCurrentPreviewFrame(frame_number);
            std::cout << "Refine " << frame_number << ", level " << level << ", step " << time_step << ", pass " << this_pass << "..." << std::flush;

            bool ret = framePtr(frame_number)->refine(level, this_pass, last_coarse_pass, time_step,
                                                      framePtr(frame_number - time_step),
                                                      framePtr(frame_number + time_step),
                                                      init_mode);
            if (!ret)
                return false;

            std::cout << " done." << std::endl;
        }
    }

    // If just doing odd, just copy forward the even frames (to keep indexing consistent)
    if (op == REFINE_ODD) {
        for (int frame = _firstFrame; frame <= _lastFrame; frame += coarse_step) {
            std::cout << "Link even frame, pass " << framePtr(frame)->lastSavedPass(level)
                      << " to " << this_pass << ", frame " << frame << "..." << std::flush;
            //_cudaWorkingBuffers->linkTempImagesToNextPass(level, frame, source_pass, this_pass);
            framePtr(frame)->linkImagesToPass(level, this_pass);
            std::cout << " done." << std::endl;
        }
    } else if (op == REFINE_EVEN) {
        for (int frame = _firstFrame + time_step; frame <= _lastFrame; frame += coarse_step) {
            std::cout << "Link odd frame, pass " << framePtr(frame)->lastSavedPass(level)
                      << " to " << this_pass << ", frame " << frame << "..." << std::flush;
            //_cudaWorkingBuffers->linkTempImagesToNextPass(level, frame, source_pass, this_pass);
            framePtr(frame)->linkImagesToPass(level, this_pass);
            std::cout << " done." << std::endl;
        }
    }

    return true;
}



bool SynthesisProcessor::renderAnimationTCTF()
{
    bool do_batch = !(_data->getBoolParameter("run_local"));
    bool run_offline = _data->getBoolParameter("run_offline");

    int max_spatial_level = _levels - 1;
    int max_time_step = std::max(1, _data->getMaxRibbonStep());

    int spatial_level = max_spatial_level;
    int time_step = max_time_step;

    QList<SynthesisOperation> schedule_ops;
    QList<int> schedule_levels;
    QList<int> schedule_steps;

    if (_keyFrameInterpolation) {
        // If we have keyframes, just do an initialization pass with full fixup.
        schedule_ops.push_back(INITIALIZE_EVEN_AND_ODD);
    } else {
        // If there are no keyframes, we make some "keyframes"
        // with a consecutive pass.
        schedule_ops.push_back(CONSECUTIVE_PASS);
    }
    schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);

    // TCTF with space as inner loop:
    for (; spatial_level >= _finalLevel; spatial_level--) {
        schedule_ops.push_back(REFINE_EVEN_AND_ODD); schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);
        if (time_step < _lastFrame - _firstFrame) {
            for (int i = 0; i < _fixupPasses + 1; i++) {
                schedule_ops.push_back(REFINE_EVEN_AND_ODD); schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);
            }
        }
    }

    for (time_step = max_time_step / 2; time_step > 0; time_step /= 2) {
        for (spatial_level = max_spatial_level; spatial_level >= _finalLevel; spatial_level--) {
            schedule_ops.push_back(REFINE_ODD); schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);
            if (time_step < _lastFrame - _firstFrame) {
                for (int i = 0; i < _fixupPasses; i++) {
                    schedule_ops.push_back(REFINE_EVEN_AND_ODD); schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);
                }
            }
        }
    }


    // TCTF with time as inner loop:
    /*
    time_step /= 2;

    while (time_step >= 1) { //(1 << spatial_level)) {
        schedule_ops.push_back(REFINE_ODD); schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);
        if (_fixupPasses > 0) {
            schedule_ops.push_back(REFINE_EVEN); schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);
            for (int i = 1; i < _fixupPasses; i++) {
                schedule_ops.push_back(REFINE_EVEN_AND_ODD); schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);
            }
        }
        time_step /= 2;
    }
    spatial_level--;

    // Now refine each spatial level, moving coarse to fine in time at each level.
    while (spatial_level >= _finalLevel) {
        time_step = max_time_step;
        schedule_ops.push_back(INITIALIZE_EVEN_AND_ODD); schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);
        time_step /= 2;
        while (time_step >= 1) { //(1 << spatial_level)) {
            schedule_ops.push_back(REFINE_ODD); schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);
            if (spatial_level > 1) {
                if (_fixupPasses > 0) {
                    schedule_ops.push_back(REFINE_EVEN); schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);
                    for (int i = 1; i < _fixupPasses; i++) {
                        schedule_ops.push_back(REFINE_EVEN_AND_ODD); schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);
                    }
                }
            }
            time_step /= 2;
        }
        spatial_level--;
    }
    */

    // Finally, just upsample any remaining levels down to the bottom of the pyramid.
    time_step = 1;
    while (spatial_level >= 0) {
        schedule_ops.push_back(SPATIAL_UPSAMPLE); schedule_levels.push_back(spatial_level); schedule_steps.push_back(time_step);
        spatial_level--;
    }

    _cudaWorkingBuffers->resetAdvectedBuffers();
    _cudaWorkingBuffers->setCurrentLevel(spatial_level);

    int passes_this_level = 0;
    int passes_last_level = 0;

    if (do_batch) {
        _work_schedule = new WorkSchedule();
    }

    // Now follow the schedule of upsampling and fixup.
    int last_spatial_level = -1;
    for (int step = 0; step < schedule_ops.size(); step++) {
        spatial_level = schedule_levels[step];
        time_step = schedule_steps[step];
        SynthesisOperation op = schedule_ops[step];

        _data->setCurrentPreviewPass(passes_this_level);
        _data->setCurrentPreviewLevel(spatial_level);
        _cudaWorkingBuffers->setCurrentLevel(spatial_level);

        if (last_spatial_level != spatial_level) {
            _style.load(spatial_level);
            passes_last_level = passes_this_level;
            last_spatial_level = spatial_level;
        }

        if (op == INITIALIZE_EVEN_AND_ODD) {
            if (!renderInitializationPass(spatial_level, time_step, passes_this_level-1))
                return false;
            passes_this_level = 1;
        } else if (op == REFINE_ODD ||
                   op == REFINE_EVEN ||
                   op == REFINE_EVEN_AND_ODD) {
            if (!renderRefinePass(spatial_level, passes_last_level-1, passes_this_level, time_step, op))
                return false;
            passes_this_level++;
        } else if (op == SPATIAL_UPSAMPLE) {
            if (!renderSpatialUpsample(spatial_level+1, passes_this_level-1))
                return false;
            passes_this_level = 1;
        } else if (op == CONSECUTIVE_PASS) {
            if (!renderConsecutivePass(spatial_level, passes_this_level, passes_this_level-1, time_step))
                return false;
            passes_this_level++;
        }
    }

    if (do_batch) {
        _work_schedule->addCopyOutputToFinal(_firstFrame, _lastFrame, passes_this_level-1);

        _work_schedule->setWorkingDir(_data->getTemporaryDir());

        if (run_offline) {
            _work_schedule->spoolTinaJob();
        } else {
            BatchProcessor bp(this);
            bp.runSchedule(*_work_schedule);
        }

        setPreserveImagePyramidCache(true);

        delete _work_schedule;
        _work_schedule = NULL;
    } else {
        copyOutputToFinal(passes_this_level-1);
    }

    return true;
}

float SynthesisProcessor::updateProgress(int level, int pass, int frames_this_pass)
{
    float total_work = 0;
    int num_frames = _lastFrame - _firstFrame + 1;
    int num_passes = _fixupPasses + 1;

    for (int i = _levels-1; i >= _finalLevel; i--) {
        total_work += (float)num_frames * (float)num_passes / (float)(1 << i);
    }
    float finished_work = 0;
    for (int i = _levels-1; i >= _finalLevel; i--) {
        if (i > level) {
            finished_work += (float)num_frames * (float)num_passes / (float)(1 << i);
        } else if (i == level) {
            finished_work += (float)num_frames * (float)pass / (float)(1 << i);
            finished_work += (float)frames_this_pass / (float)(1 << i);
        }
    }

    float percent = (finished_work / total_work) * 100;

    if (_progress) {
        _progress->setValue(percent);
    }
    if (_print_progress) {
        cout << "ALF_PROGRESS " << (int)percent << "%\n";
    }
    return percent;
}



Color4 SynthesisProcessor::outputPixel(int x, int y)
{
    return _cudaWorkingBuffers->outputPixel(x,y);
}

float4 SynthesisProcessor::residualPixel(int x, int y)
{
    return _cudaWorkingBuffers->residualPixel(x,y);
}

PatchXF SynthesisProcessor::offsetsPixel(int x, int y)
{
    return _cudaWorkingBuffers->offsetsPixel(x,y);
}

float SynthesisProcessor::histogramPixel(int x, int y)
{
    return _cudaWorkingBuffers->histogramPixel(x,y);
}



void SynthesisProcessor::si_examineCuda()
{
    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("Cuda driver version %d, runtime version %d\n", driverVersion, runtimeVersion);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        if (device == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                printf("There is 1 device supporting CUDA\n");
            else
                printf("There are %d devices supporting CUDA\n",
                       deviceCount);
        }
    }

    // Only supported on Cuda 4.0+
    if (deviceCount > 1) {
        bool allAccess = true;
        for (int i = 0; i < deviceCount; i++) {
            for (int j = i+1; j < deviceCount; j++) {
                int canAccessPeer;
                cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                allAccess = allAccess && canAccessPeer;
            }
        }
        if (allAccess) {
            printf("All cuda devices can access each other.\n");
        } else {
            printf("Cuda devices cannot access each other.\n");
        }
    }
    std::flush(cout);
}
