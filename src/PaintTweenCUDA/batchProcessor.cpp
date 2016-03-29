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

#include "batchProcessor.h"
#include "workingBuffers.h"

#include "texSynth_kernel.h"

#include "imageIO.h"
#include "stats.h"

#include <QDebug>
#include <QDir>
#include <iostream>

BatchProcessor::BatchProcessor(SynthesisProcessor* parent)
{
    _parent = parent;
    _style = parent->style();
    _working_buffers = parent->workingBuffers();
    _thread_data = parent->threadData();
}

BatchProcessor::~BatchProcessor()
{
    if (_parent) {
        // Our pointers are owned by our parent.
        return;
    }
}

bool BatchProcessor::runSchedule(const WorkSchedule& schedule, const QList<int>* work_units)
{
    __TIME_CODE_BLOCK("BatchProcessor::runSchedule");

    _parent->si_offline_initSynthesis();
    setPreserveImagePyramidCache(true);

    TexSynth::setup(_working_buffers->params());

    QList<int> all_units;
    if (work_units == NULL) {
        for (int i = 0; i < schedule.numWorkUnits(); i++) {
            all_units.push_back(i);
        }
        work_units = &all_units;
    }

    if (DataAccess::instance().getBoolParameter("run_offline") == false) {
        // Set single step mode if running online.
        _thread_data->_synthesisState = STATE_WORKING;
    }

    foreach(int task_idx, *work_units) {
        const WorkUnit& task = schedule.workUnit(task_idx);
        int frame_number = task.frame();
        int level = task.level();
        int pass = task.pass();

        DataAccess::instance().setCurrentPreviewFrame(frame_number);
        DataAccess::instance().setCurrentPreviewLevel(level);
        DataAccess::instance().setCurrentPreviewPass(pass);

        SynthesisFrame* this_frame = framePtr(frame_number);

        if (task.type() == "spatial_upsample") {

            int coarse_level = task.arg("coarse_level").toInt();
            int fine_level = coarse_level - 1;
            int source_pass = task.arg("source_pass").toInt();

            std::cout << "Spatial upsample, pass " << source_pass + 1 << ", level " << coarse_level << " to " << fine_level << ", frame " << frame_number << "...";

            _working_buffers->setCurrentLevel(fine_level);
            bool ret = _style->load(fine_level);
            ret = ret && framePtr(frame_number)->spatialUpsample(coarse_level, fine_level, source_pass);
            if (!ret)
                return false;

            std::cout << " done.\n" << std::flush;

        }
        else if (task.type() == "refine") {

            int last_coarse_pass = task.arg("last_coarse_pass").toInt();
            int time_step = task.arg("time_step").toInt();
            int op_flags = task.arg("op_flags").toInt();

            std::cout << "Refine " << frame_number << ", level " << level << ", step " << time_step << ", pass " << pass << "..." << std::flush;

            SynthesisFrame* prior_frame = framePtr(frame_number - time_step);
            SynthesisFrame* next_frame = framePtr(frame_number + time_step);

            _working_buffers->setCurrentLevel(level);
            _style->load(level);
            this_frame->createRandomPyramid(level);

            bool ret = this_frame->refine(level, pass, last_coarse_pass, time_step,
                                          prior_frame, next_frame, op_flags);
            if (!ret)
                return false;

            std::cout << " done." << std::endl;

        }
        else if (task.type() == "initialize") {

            int source_pass = task.arg("source_pass").toInt();
            int time_step = task.arg("time_step").toInt();
            int op_flags = task.arg("op_flags").toInt();

            std::cout << "Initialize " << frame_number << ", level " << level << ", step " << time_step << "..." << std::flush;

            _working_buffers->setCurrentLevel(level);
            _style->load(level);
            this_frame->createRandomPyramid(level);

            bool ret = this_frame->refine(level, 0, source_pass, time_step, 0, 0, op_flags);
            if (!ret)
                return false;

            std::cout << " done." << std::endl;

        }
        else if (task.type() == "copy_output_to_final") {

            int first_frame = task.arg("first_frame").toInt();
            int last_frame = task.arg("last_frame").toInt();
            int final_pass = task.arg("final_pass").toInt();

            std::cout << "Copying frames " << first_frame << "-" << last_frame << " to final location..." << std::flush;

            copyOutputToFinal(first_frame,
                              last_frame,
                              final_pass);

            // This is the last thing we do, so it's ok to clean up the cache images.
            setPreserveImagePyramidCache(false);

            std::cout << " done." << std::endl;
        }
    }

    return true;
}

bool BatchProcessor::runSchedule(const WorkSchedule& schedule, const QString& task_string)
{
    if (task_string.isEmpty()) {
        return runSchedule(schedule);
    } else {
        QList<int> tasks;
        QStringList tokens = task_string.split(",");
        for (int i = 0; i < tokens.size(); i++) {
            if (!tokens[i].isEmpty()) {
                tasks.push_back(tokens[i].toInt());
            }
        }
        return runSchedule(schedule, &tasks);
    }
}

void BatchProcessor::copyOutputToFinal(int first_frame, int last_frame, int final_pass)
{
    DataAccess& data = DataAccess::instance();
    QDir out_dir = QFileInfo(ImageIO::parsePath(data.getOutPath(OUT_OUTPUT_FINAL).arg(""),0)).dir();
    for (int frame = first_frame; frame <= last_frame; frame++) {
        QString temp_path = ImageIO::temporaryOutputPath(data.getOutPath(OUT_OUTPUT),0,frame,final_pass,0);
        QString final_path = ImageIO::parsePath(data.getOutPath(OUT_OUTPUT_FINAL).arg(""),frame);

        QFile temp_file(temp_path);
        if (!temp_file.exists()) {
            qWarning("Could not find %s", qPrintable(temp_path));
            continue;
        }

        out_dir.remove(final_path);
        temp_file.copy(final_path);
    }

}

SynthesisFrame* BatchProcessor::framePtr(int frame_number)
{
    if (_parent) {
        return _parent->framePtr(frame_number);
    } else {
        assert(0);
    }
}
