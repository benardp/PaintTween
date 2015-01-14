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

#ifndef BATCHPROCESSOR_H
#define BATCHPROCESSOR_H

#include "types.h"
#include <vector>
#include <string>

#include "workSchedule.h"
#include "synthesisProcessor.h"

#include <QtCore/QHash>
#include <QtCore/QList>

class Style;
class SynthesisFrame;
class WorkingBuffers;

class BatchProcessor
{
public:
    BatchProcessor(SynthesisProcessor* parent);
    ~BatchProcessor();

    bool runSchedule(const WorkSchedule& schedule, const QList<int>* work_units = 0);
    bool runSchedule(const WorkSchedule& schedule, const QString& task_string);

protected:
    void copyOutputToFinal(int first_frame, int last_frame, int final_pass);
    SynthesisFrame* framePtr(int frame_number);

protected:
    SynthesisProcessor* _parent;

    Style* _style;

    QHash<int, SynthesisFrame*> _frames;
    WorkingBuffers* _working_buffers;

    SynthesisThreadData* _thread_data;
};

#endif
