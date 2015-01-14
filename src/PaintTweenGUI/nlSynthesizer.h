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

#ifndef NLSYNTHESIZER_H
#define NLSYNTHESIZER_H

#include <QtCore/QDebug>
#include <QtCore/QTimer>

#include "PaintTweenCUDA/synthesisProcessor.h"
#include "PaintTweenCUDA/workingBuffers.h"
#include "PaintTweenCUDA/progressCallback.h"
#include "nlDataAccess.h"

class NLMainWindow;

class NLSynthesizer : public QObject, public NLDataAccess
{
    Q_OBJECT

public:
    void initialize(int cudaDevice);
    void initialize(NLMainWindow* mainWindow, int cudaDevice);
    ~NLSynthesizer();

    void finalCleanUp();

    bool readWorkingSet(const QString &filename, bool styleRefreshOnly);
    bool writeWorkingSet(const QString &filename);

    WorkingBuffers* workingBuffers() { return _synthesisProcessor->workingBuffers(); }
    SynthesisProcessor* synthesisProcessor() { return _synthesisProcessor; }

    int currentLevel() const;
    int currentStyleWidth(int styleIndex) const;
    int currentStyleHeight(int styleIndex) const;

    //---------------------------------------------------------
    // Still image synthesis:
    bool updateStillSynthesis();

    inline bool isRealtimeSynthesisRunning() const {return _rtSynthesisRunning; }
    inline bool isRealtimeSynthesisPaused() const {return _rtSynthesisRunning && !_timer->isActive(); }

    bool updateCurrentFrame(int i);
    bool updateViewerLevel(int i);
    bool updateViewerPass(int i);

    //---------------------------------------------------------
    // Animation Synthesis:
    bool runAnimationSynthesis();
    void setAnimationProgress(ProgressCallback* progress) { _animationProgress = progress; }

    bool copyToGL(struct cudaGraphicsResource* destination, InputElements element);
    bool copyToGL(struct cudaGraphicsResource* destination, int styleNum, StyleElements element);

    static NLSynthesizer& instance() { return _instance; }

public slots:
    void advanceSynthesis();
    void updateSynthesis();
    void pauseSynthesis();
    void resumeSynthesis();
    void cancelSynthesis();
    void singleStepSynthesis();
    void singleStepBackSynthesis();

    void updateFirstFrame(int i);
    void updateLastFrame(int i);
    void updateRealtimeSynthesisMode(int i);
    void updateParameters();

    void examineCuda();

signals:
    void animationDone();
    void synthesisStarted();
    void synthesisAdvanced();
    void parameterChanged();
    void cleaningUp();

protected:
    NLSynthesizer() {};

    // Start/Stop display loop:
    void startDisplay();
    void stopDisplay();

    void printMessage(const QString& msg) const;

private:
    NLMainWindow* _mainWindow;

    SynthesisProcessor* _synthesisProcessor;

    ProgressCallback* _animationProgress;

    // Timer for realtime synthesis:
    QTimer* _timer;

    bool _rtSynthesisRunning;
    bool _rtSynthesisHasBeenPaused;

    static NLSynthesizer _instance;
};

#endif // NLSYNTHESIZER_H
