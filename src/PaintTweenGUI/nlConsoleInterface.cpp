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

#include "nlConsoleInterface.h"

#include "PaintTweenCUDA/synthesisProcessor.h"
#include "PaintTweenCUDA/batchProcessor.h"
#include "PaintTweenCUDA/workSchedule.h"
#include "PaintTweenCUDA/progressCallback.h"
#include "PaintTweenCUDA/stats.h"

NLConsoleInterface::NLConsoleInterface(int cudaDevice)
    : _dataLoaded(false)
{
    _synthesizer = &NLSynthesizer::instance();
    _synthesizer->initialize(cudaDevice);
    _progress = new ProgressCallback();
    _schedule = NULL;
}

NLConsoleInterface::~NLConsoleInterface()
{
    // Delete aux. data:
    delete _progress;
    delete _schedule;
}

// Function to perform final cleanup, which makes sure that all
// CUDA data is deallocated, and thread exist is called. Must be
// called before destructor is called!
void NLConsoleInterface::finalCleanUp()
{
    _synthesizer->finalCleanUp();
    cudaThreadExit();

    qDebug() << Stats::instance().allTimerStatistics();
}

// Function to load and setup data:
bool NLConsoleInterface::setup(const QString& appPath,
                               const QString& fileName, const QString& outputDir,
                               const QString& workingDir, const QString& scheduleFile,
                               const QString& mode, const QString& taskString,
                               int frame)
{
    __TIME_CODE_BLOCK("NLConsoleInterface::setup");

    bool success = _synthesizer->readWorkingSet(fileName,false);
    if (mode == "batch") {
        _schedule = new WorkSchedule;
        success = success && _schedule->load(scheduleFile);
        _taskString = taskString;
    }
    if(success){
        _dataLoaded = true;
        _synthesizer->setOutputDir(outputDir);
        _synthesizer->setStringParameter("app_path", appPath);
        if (workingDir.isEmpty()) {
            _synthesizer->setStoreIntermediateImagesInTemp(false);
        } else {
            _synthesizer->setTemporaryDir(workingDir);
            _synthesizer->setStoreIntermediateImagesInTemp(true);
        }
        if(mode == "still" && frame <= _synthesizer->lastFrame() && frame >= _synthesizer->firstFrame()){
            _synthesizer->setFirstFrame(frame);
            _synthesizer->setLastFrame(frame);
        }
    }
    return success;
}

bool NLConsoleInterface::runAnimation(){
    __TIME_CODE_BLOCK("NLConsoleInterface::runAnimation");

    if (!_dataLoaded){
        qCritical() << "Data has not been loaded! ";
        return false;
    }

    if (_schedule) {
        BatchProcessor bp(_synthesizer->synthesisProcessor());
        return bp.runSchedule(*_schedule, _taskString);
    } else {
        _synthesizer->setAnimationProgress(_progress);
        return _synthesizer->runAnimationSynthesis();
    }

}
