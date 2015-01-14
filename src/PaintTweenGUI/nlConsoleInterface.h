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

#ifndef NLCONSOLEINTERFACE_H
#define NLCONSOLEINTERFACE_H

#include <QtCore/QString>

#include "nlSynthesizer.h"

class ProgressCallback;
class WorkSchedule;

class NLConsoleInterface
{

public:
    NLConsoleInterface(int cudaDevice);
    ~NLConsoleInterface();

    // Function to perform final cleanup, which makes sure that all
    // CUDA data is deallocated, and thread exit is called.
    void finalCleanUp();

    bool setup(const QString& appPath,
               const QString& fileName, const QString& outputDir,
               const QString& workingDir, const QString& scheduleFile,
               const QString& mode, const QString& taskString,
               int frame);
    bool runAnimation();

private:
    NLSynthesizer* _synthesizer;

    WorkSchedule* _schedule;
    QString _taskString;

    // Progress callback:
    ProgressCallback* _progress;

    // Status flag:
    bool _dataLoaded;
};

#endif
