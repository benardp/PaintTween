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

#include "nlAnimationDialog.h"
#include "nlSynthesizer.h"

#include <QtGui/QPushButton>
#include <QtGui/QProgressBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QFileDialog>

#include <iostream>

NLAnimationDialog::NLAnimationDialog(QWidget* parent,
                                     NLSynthesizer* synthesizer,
                                     QString working)
    : QDialog(parent),
      _currentStep(0),
      _currentFrame(0),
      _progressBar(NULL),
      _synthesizer(synthesizer),
      _workingDir(working)
{
    this->setWindowTitle("Animation synthesis");

    // Create buttons:
    _run = new QPushButton(tr("Run"), this);
    connect(_run, SIGNAL(clicked()), this, SLOT(run()));

    // Create progress bar:
    _progressBar = new QProgressBar(this);
    _progressBar->setMinimum(0);
    _progressBar->setMaximum(100);
    _progressBar->setValue(0);

    // Create layout, add widgets and set layout:
    _layout = new QVBoxLayout(this);
    _layout->addWidget(_run);
    _layout->addWidget(_progressBar);
    setLayout(_layout);
}

// Run the animation (image sequence) synthesis:
void NLAnimationDialog::run(void){
    if (_synthesizer){
        // Reset the progressbar:
        _progressBar->reset();
        resetSteps();

        // Set the progress range (which is calculated using
        // the number of levels, frames and fixup passes):

        int levels = _synthesizer->getIntParameter("levels") - _synthesizer->getIntParameter("maxLevel");
        int frames = _synthesizer->lastFrame() - _synthesizer->firstFrame() + 1;
        int passes = _synthesizer->getIntParameter("fixupPasses");
        int totFrames = (passes == 0) ? frames : (passes*(frames-1) + frames);
        _progressBar->setMaximum(levels*totFrames);

        // Finally set the progressbar for callback:
        _synthesizer->setAnimationProgress(this);

        QString dirName;
        if (_workingDir.isEmpty()) {
            dirName = QFileDialog::getExistingDirectory(
                        this,
                        tr("Choose output directory"), QDir::currentPath());
        } else {
            dirName = _workingDir;
        }
        if (!dirName.isEmpty()){
            update();
            _synthesizer->setOutputDir(dirName);
            _synthesizer->setStoreIntermediateImagesInTemp(false);

            _synthesizer->runAnimationSynthesis();
        }
        setProgressBar(NULL);
    }
}
