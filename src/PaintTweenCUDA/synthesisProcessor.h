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

#ifndef SYNTHESISPROCESSOR_H
#define SYNTHESISPROCESSOR_H

#include "types.h"
#include <vector>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "imagePyramid.h"
#include "dataAccess.h"
#include "style.h"
#include "synthesisFrame.h"

#include <QtCore/QThread>
#include <QtCore/QMutex>
#include <QtCore/QWaitCondition>


class ProgressCallback;
class DataAccess;
class WorkingBuffers;
class WorkSchedule;

// Constants to define the states of the algorithm.
typedef enum {
    STATE_READY = 0,
    STATE_WORKING,
    STATE_WAITING_FOR_CONTINUE,
    STATE_CONTINUING,
    STATE_FINISHED,
    STATE_ABORTED,
    STATE_TERMINATING,
    STATE_RUNNING_NO_WAITING,
    STATE_INVALID
} SynthesisState;


typedef enum {
    INITIALIZE_EVEN_AND_ODD,
    REFINE_ODD,
    REFINE_EVEN,
    REFINE_EVEN_AND_ODD,
    SPATIAL_UPSAMPLE,
    CONSECUTIVE_PASS
} SynthesisOperation;

class SynthesisThreadData
{
public:
    QMutex  _mutex;
    QWaitCondition _waiter;
    QString _waitMessage;
    SynthesisState _synthesisState;

    // returns false if the thread should exit.
    bool waitForContinue(const char* msg);
};


class SynthesisProcessor 
{

public:
    SynthesisProcessor(DataAccess* data);
    ~SynthesisProcessor();

    // Function to perform a next step in the synthesis.
    // Returns true if still in processing stage, otherwise false:
    bool si_realtime_advanceSynthesis(bool take_snapshots);
    bool si_realtime_backtrackSynthesis();

    void si_realtime_initSynthesis(bool cleanStyle, bool cleanImage);
    void si_realtime_startSynthesis();
    void si_realtime_restartSynthesis();
    void si_cancelSynthesis();
    void si_cleanupSynthesis();
    void si_parameterChanged();

    void si_offline_initSynthesis();
    void si_offline_startSynthesis(ProgressCallback* progress);

    void si_examineCuda();

    Color4 outputPixel(int x, int y);
    PatchXF offsetsPixel(int x, int y);
    float4 residualPixel(int x, int y);
    float histogramPixel(int x, int y);
    Style* style() { return &_style; }
    SynthesisFrame* framePtr(int frame_number);

    void si_saveImage(const QString& filePath);

    WorkingBuffers* workingBuffers() { return _cudaWorkingBuffers; }
    const QString& waitMessage() { return _thread_state._waitMessage; }
    SynthesisThreadData* threadData() { return &_thread_state; }

    bool copyInputArray(InputElements element, cudaArray *array);
    bool copyStyleArray(StyleElements element, int numStyle, cudaArray *array);

protected:
    //--------------------------------------------------------------------------------------
    // Global parameters:

    // Number of iterations globally and on the first frame: (That is the
    // actual number of propagate/scatter steps do to per scalespace level:
    int _iterations;

    // number of iterations to do to the first frame of the animation -
    // you should typically do more so that the first frame stabalizes,
    // or else you will see it 'stabalize' over the first few frames of the anim,
    // which looks bad:
    int _iterationsFirstFrame;

    // Number of levels:
    int _levels;
    // Stop the synthesis at this level
    int _finalLevel;

    bool _renderPreviewStill;

    // If true, linearly interpolate between two key-frames and use this interpolation
    // as guide fro the synthesis
    bool _keyFrameInterpolation;

    // First and last frame (inclusive) number:
    int _firstFrame;
    int _lastFrame;
    int _storageFirstFrame;

    // Number of extra fixup passes:
    int _fixupPasses;

    // If 0, first pass is forward
    // If 1, first pass is backward
    PassDirection _firstPassDirection;

    // If true, write the keyframe preprocess results to the output directory.
    bool _savePreprocessToOutput;
    bool _preprocessed;
    
    DataAccess* _data;

    Style _style;

    QVector< SynthesisFrame > _frames;
    WorkingBuffers* _cudaWorkingBuffers;

    bool accumulateRibbon(bool timeIsForwards, int width, int height);
    bool keyFramesPreprocess();

    bool renderConsecutivePass(int level, int pass, int source_pass, int time_step);
    bool renderInterpolatedFixupPass(int level, int pass, int time_step, int which_frames);
    bool renderInitializationPass(int level, int time_step, int source_pass);
    bool renderRefinePass(int level, int last_coarse_pass, int this_pass, int time_step, SynthesisOperation op);
    bool renderSpatialUpsample(int coarse_level, int source_pass);

    void updateParameters(bool for_offline_synthesis, bool load_default);
    void initializeFrameStorage();
    void copyOutputToFinal(int final_pass);

    bool runBatch();
    bool renderAnimation();
    bool renderAnimationFMBM();
    bool renderAnimationFBMO();
    bool renderAnimationTCTF();
    bool renderAnimationIND();

    float updateProgress(int level, int pass, int frame);

    // Multi-threaded synthesis implementation members
    class Worker : public QThread {
    public:
        Worker(SynthesisProcessor* parent) : _parent(parent), _stages(0) {}
        void shutDown();
        void runStages(int stages);

        static void sleep(unsigned long secs){ QThread::sleep(secs); }
    protected:
        void run();
    protected:
        SynthesisProcessor* _parent;
        int _stages;
    };


    SynthesisThreadData _thread_state;
    Worker* _workerThread;

    WorkSchedule* _work_schedule;

    ProgressCallback* _progress;
    bool _print_progress;

};

#endif
