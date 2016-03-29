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

#ifndef NLANIMATIONDIALOG_H
#define NLANIMATIONDIALOG_H

#include <QDialog>
#include <QProgressBar>

#include "PaintTweenCUDA/progressCallback.h"

class NLSynthesizer;
class NLImageSeqChooser;
class QPushButton;
class QProgressBar;
class QVBoxLayout;

//-----------------------------------------------------------------------------
//
// PaintTween Animation Dialog:
// GUI class as user interface to start synthesis of an animation
// (image sequence) and getting information about the progress of the
// synthesis.
//
//-----------------------------------------------------------------------------

class NLAnimationDialog : public QDialog, public ProgressCallback
{
    Q_OBJECT

public:
    NLAnimationDialog(QWidget* parent, NLSynthesizer* synthesizer,
                      QString working = QString());

    // Setter for progress bar:
    inline void setProgressBar(QProgressBar* bar){_progressBar = bar; }

    // Functions to reset the progress counter:
    void resetSteps(){ _currentStep = 0; }
    void resetFrames(){ _currentFrame = 0; }

    // Callback to request a new frame:
    void newFrame(int frame){
        _currentFrame = frame;
    }

    // Function for advancing a progress feedback (such as the
    // progressbar to give the user an indication on the progress):
    void newStep(){
        _currentStep++;
        if(_progressBar){
            _progressBar->setValue(_currentStep);
        }
    }

    void setValue(int value) {
        if(_progressBar){
            _progressBar->setValue(value);
        }
    }

public slots:
    void run(void);

protected:
    int _currentStep;
    int _currentFrame;

    //---------------------------------------------------------------------
    // GUI elements:
    QProgressBar* _progressBar;
    QPushButton* _run;
    QVBoxLayout* _layout;

    //---------------------------------------------------------------------
    // Control object in charge of the synthesis:
    NLSynthesizer* _synthesizer;

    QString _workingDir;
};

#endif
