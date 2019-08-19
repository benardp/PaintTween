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

#include "nlSynthesizer.h"

#ifdef _MSC_VER
#include <Windows.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <QtCore/QFile>
#include <QMessageBox>
#include <QStatusBar>
#include <QtXml/QDomDocument>
#include <QtNetwork/QHostInfo>

#include "nlMainWindow.h"

#include "PaintTweenCUDA/imagePyramid.h"

NLSynthesizer NLSynthesizer::_instance;

void NLSynthesizer::initialize(int cudaDevice)
{
    std::cout << "Initializing CUDA at device " << cudaDevice << "... " << std::flush;
    cudaSetDevice(cudaDevice);
    std::cout << "done." << std::endl;

    _synthesisProcessor = new SynthesisProcessor(this);
    _rtSynthesisRunning = false;
    _timer = NULL;
}

void NLSynthesizer::initialize(NLMainWindow* mainWindow, int cudaDevice)
{
    std::cout << "Initializing CUDA at device " << cudaDevice << "... " << std::endl;
    //cudaGLSetGLDevice(cudaDevice);
    std::cout << "done." << std::endl;

    _synthesisProcessor = new SynthesisProcessor(this);
    _rtSynthesisRunning = false;

    _mainWindow = mainWindow;

    // Create the timer:
    _timer = new QTimer(this);
    connect(_timer, SIGNAL(timeout()), this, SLOT(advanceSynthesis()));
}

NLSynthesizer::~NLSynthesizer()
{
    delete _synthesisProcessor;
    delete _timer;
}

void NLSynthesizer::finalCleanUp()
{
    // Function to perform a final cleanup, which makes sure that all
    // CUDA data is deallocated, and thread exit is called.

    if (getPreserveImagePyramidCache()) {
        qDebug() << "Final cleanup (preserving pyramid caches)";
    } else {
        qDebug() << "Final cleanup (removing pyramid caches)";
    }

    emit cleaningUp();

    _synthesisProcessor->si_cleanupSynthesis();

    cudaDeviceSynchronize();
}

bool NLSynthesizer::readWorkingSet(const QString& filename, bool styleRefreshOnly)
{
    if(!styleRefreshOnly)
        clear();

    QDomDocument wsDoc("WorkingSet");

    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly))
        return false;
    if (!wsDoc.setContent(&file)) {
        file.close();
        return false;
    }
    file.close();

    QDomElement docElem = wsDoc.documentElement();
    if(docElem.isNull()) {
        qWarning() << "Empty working set!";
        return false;
    }

    QDomElement styleElt = docElem.firstChildElement("styles");
    if(!styleElt.isNull()){
        if(!_styles.load(filename, styleElt))
            return false;
    }else{
        qWarning() << "No style data found!";
        return false;
    }

    setStringParameter("working_set_creator", docElem.attribute("creator"));
    setStringParameter("render_crews", docElem.attribute("render_crews"));

    QDomElement paramElt = docElem.firstChildElement("params");
    if(!paramElt.isNull()){
        NLParameters::instance().load(paramElt, getNumStyles());
    }

    setFirstFrame(NLParameters::instance().getInt("firstFrame"));
    setLastFrame(NLParameters::instance().getInt("lastFrame"));

    QDomElement animElt = docElem.firstChildElement("anim");
    if(!animElt.isNull()){
        if(!_shot.setup(filename, animElt, firstFrame(), lastFrame())){
            return false;
        }
        setOutputDir(getWorkingDir());
    }else{
        qWarning() << "No animation data found!";
        return false;
    }

    _loaded = true;
    setStringParameter("working_set_path", filename);

    if (!NLParameters::instance().getBool("run_offline")) {
        _synthesisProcessor->si_realtime_initSynthesis(true,true);
    }

    return true;
}

bool NLSynthesizer::writeWorkingSet(const QString &filename)
{
    QDomDocument wsDoc("WorkingSet");

    QDomElement root = wsDoc.createElement("doc");

    QString user = getenv("USER");
    QString host = QHostInfo::localHostName();
    root.setAttribute("creator", user + "@" + host);

    wsDoc.appendChild(root);

    _shot.save(wsDoc,root);
    _styles.save(wsDoc,root);
    NLParameters::instance().save(wsDoc,root);

    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)) {
        qCritical("Could not write working set %s", qPrintable(filename));
        return false;
    }

    qDebug("Writing working set to %s", qPrintable(filename));

    file.write(wsDoc.toByteArray());

    return true;
}

int NLSynthesizer::currentLevel() const
{
    return _synthesisProcessor->workingBuffers()->currentLevel();
}

int NLSynthesizer::currentStyleWidth(int styleIndex) const
{
    return _synthesisProcessor->workingBuffers()->currentStyleWidth(styleIndex);
}

int NLSynthesizer::currentStyleHeight(int styleIndex) const
{
    return _synthesisProcessor->workingBuffers()->currentStyleHeight(styleIndex);
}

void NLSynthesizer::startDisplay(){
    _timer->start(100);
}

void NLSynthesizer::stopDisplay(){
    _timer->stop();
}

bool NLSynthesizer::updateStillSynthesis()
{
    // Check if data is up to date, if false, we have to
    // potentially transfer new images to CUDA, and we have
    // to cleanup and initialize CUDA and CUDA<->OpenGL:
    if (!_dataUptodate){
        qDebug() << "Data has been updated, restart! ";

        // Check if all the necessary data has been provided:
        QString msg;
        if (!isStyleReady(msg)){
            printMessage(msg);
            return false;
        }

        if (!isAnimReady(msg)){
            printMessage(msg);
            return false;
        }

        // If we have a synthesis running, we first have to stop this synthesis:
        if (_rtSynthesisRunning){
            // First stop the display loop, this will make sure that
            // the synthesis widget stops displaying the synthesis:
            this->stopDisplay();

            // Reset running flag:
            _rtSynthesisRunning = false;
        }

        // Set flag to up to date:
        _dataUptodate = true;

        // Start synthesis:
        _synthesisProcessor->si_realtime_initSynthesis(false,true);
        _synthesisProcessor->si_realtime_startSynthesis();

    } else {
        std::cout << "Data is up to date! " << std::endl;

        // If we have a synthesis running we stop it:
        if (_rtSynthesisRunning){
            // First stop the display loop, this will make sure that
            // the synthesis widget stops displaying the synthesis:
            this->stopDisplay();

            // Reset running flag:
            _rtSynthesisRunning = false;
        }

        // Tell CUDA that parameters have potentially changed,
        // so CUDA starts the synthesis from the beginning:
        _synthesisProcessor->si_realtime_restartSynthesis();

    }
    // Start synthesis display loop:
    this->startDisplay();

    // Set running flag:
    _rtSynthesisRunning = true;
    _rtSynthesisHasBeenPaused = false;

    // Notify any listeners (such as mainWindow)
    emit synthesisStarted();
    emit synthesisAdvanced();

    return true;
}

void NLSynthesizer::updateSynthesis()
{
    updateStillSynthesis();
    _mainWindow->statusBar()->showMessage(tr("Synthesis started"));
    _mainWindow->updateSynthesisControls();
}

void NLSynthesizer::pauseSynthesis()
{
    if (_rtSynthesisRunning){
        stopDisplay();
        _rtSynthesisHasBeenPaused = true;
    }
    _mainWindow->statusBar()->showMessage(tr("Synthesis paused"));
    _mainWindow->updateSynthesisControls();
}

void NLSynthesizer::resumeSynthesis()
{
    if (_rtSynthesisRunning){
        startDisplay();
    }
    _mainWindow->statusBar()->showMessage(tr("Synthesis resumed"));
    _mainWindow->updateSynthesisControls();
}

void NLSynthesizer::cancelSynthesis()
{
    if (_rtSynthesisRunning){
        _synthesisProcessor->si_cancelSynthesis();
        stopDisplay();
        _rtSynthesisRunning = false;
        qDebug("Synthesis canceled.");
        _mainWindow->statusBar()->showMessage(tr("Synthesis canceled"));
    }
}

void NLSynthesizer::singleStepSynthesis()
{
    if (_rtSynthesisHasBeenPaused) {
        advanceSynthesis();
    } else {
        updateStillSynthesis();
        pauseSynthesis();
    }
    _mainWindow->statusBar()->showMessage(tr("Step forward"));
    _mainWindow->updateSynthesisControls();
}

void NLSynthesizer::singleStepBackSynthesis()
{
    if (_rtSynthesisHasBeenPaused) {
        bool success = _synthesisProcessor->si_realtime_backtrackSynthesis();

        if (success) {
            emit synthesisAdvanced();
        }
    }
    _mainWindow->statusBar()->showMessage(tr("Step back"));
    _mainWindow->updateSynthesisControls();
}

void NLSynthesizer::updateFirstFrame(int i)
{
    if (firstFrame() != i){
        setFirstFrame(i);
    }
}

void NLSynthesizer::updateLastFrame(int i)
{
    if (lastFrame() != i){
        setLastFrame(i);
    }
}

bool NLSynthesizer::updateCurrentFrame(int i)
{
    bool reloadImage = (i != _curPreviewFrame);

    bool success = goToFrame(i);
    if(success) {
        cancelSynthesis();
        _synthesisProcessor->si_realtime_initSynthesis(false,reloadImage);
        workingBuffers()->loadTempImages(getCurrentPreviewLevel(), getCurrentPreviewFrame(), getCurrentPreviewPass());
        _dataUptodate = false;
        emit parameterChanged();
    }
    return success;
}

bool NLSynthesizer::updateViewerLevel(int i)
{
    _curPreviewLevel = i;
    return updateCurrentFrame(_curPreviewFrame);
}

bool NLSynthesizer::updateViewerPass(int i)
{
    _curPreviewPass = i;
    return updateCurrentFrame(_curPreviewFrame);
}

void NLSynthesizer::updateRealtimeSynthesisMode(int i)
{
    _realtimeSynthesisMode = i;
    _dataUptodate = false;
}

void NLSynthesizer::updateParameters()
{
    _synthesisProcessor->si_parameterChanged();
    emit parameterChanged();
}

void NLSynthesizer::examineCuda()
{
    _synthesisProcessor->si_examineCuda();
}

bool NLSynthesizer::runAnimationSynthesis()
{
    _synthesisProcessor->si_offline_startSynthesis(_animationProgress);

    emit animationDone();

    return true;
}

bool NLSynthesizer::copyToGL(cudaGraphicsResource *destination, InputElements element)
{
    cudaArray* d_result;

    cudaError_t error = cudaGraphicsMapResources(1, &destination, 0);
    if (error != cudaSuccess) {std::cerr << "ERROR! GraphicsMapResources" << std::endl; }
    assert(error == cudaSuccess);

    error = cudaGraphicsSubResourceGetMappedArray(&d_result, destination, 0, 0);
    if (error != cudaSuccess) {std::cerr << "ERROR! GraphicsSubResourceGetMappedArray" << std::endl; }
    assert(error == cudaSuccess);

    _synthesisProcessor->copyInputArray(element,d_result);

    error = cudaGraphicsUnmapResources(1, &destination, 0);
    if (error != cudaSuccess) {std::cerr << "ERROR! GraphicsUnmapResources" << std::endl; }
    assert(error == cudaSuccess);

    return true;
}

bool NLSynthesizer::copyToGL(cudaGraphicsResource *destination, int styleNum, StyleElements element)
{
    cudaArray* d_result;

    cudaError_t error = cudaGraphicsMapResources(1, &destination, 0);
    if (error != cudaSuccess) {std::cerr << "ERROR! GraphicsMapResources" << std::endl; }
    assert(error == cudaSuccess);

    error = cudaGraphicsSubResourceGetMappedArray(&d_result, destination, 0, 0);
    if (error != cudaSuccess) {std::cerr << "ERROR! GraphicsSubResourceGetMappedArray" << std::endl; }
    assert(error == cudaSuccess);

    _synthesisProcessor->copyStyleArray(element,styleNum,d_result);

    error = cudaGraphicsUnmapResources(1, &destination, 0);
    if (error != cudaSuccess) {std::cerr << "ERROR! GraphicsUnmapResources" << std::endl; }
    assert(error == cudaSuccess);

    return true;
}

void NLSynthesizer::advanceSynthesis(){
    // Ask to save snapshots if we have paused before
    // (i.e., someone is paying attention to the synthesis).
    bool progress = _synthesisProcessor->si_realtime_advanceSynthesis(_rtSynthesisHasBeenPaused);

    if (_rtSynthesisRunning && !progress){
        stopDisplay();
        _rtSynthesisRunning = false;
        qDebug("Synthesis finished.\n");
        _mainWindow->statusBar()->showMessage(tr("Synthesis finished"));

        emit animationDone();
    } else {
        emit synthesisAdvanced();
    }
}

void NLSynthesizer::printMessage(const QString& msg) const
{
    if(_mainWindow)
        QMessageBox::warning(0, tr("PaintTween"), msg);
    else
        qCritical(qPrintable(msg));
}
