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

#ifndef WORKSCHEDULE_H
#define WORKSCHEDULE_H

#include "types.h"
#include <vector>
#include <string>

#include <QtCore/QHash>
#include <QtCore/QList>
#include <QtXml/QDomElement>
#include <QtCore/QVariant>

class WorkUnit
{
public:
    WorkUnit(int level, int pass, int frame);
    WorkUnit(const QDomElement& elem);

    void save(QDomElement& elem);

    int level() const { assert(_arguments.contains("level")); return _arguments["level"].toInt(); }
    int pass() const { assert(_arguments.contains("pass")); return _arguments["pass"].toInt(); }
    int frame() const { assert(_arguments.contains("frame")); return _arguments["frame"].toInt(); }

    QString type() const { assert(_arguments.contains("type")); return _arguments["type"].toString(); }
    QVariant arg(const QString& key) const { assert(_arguments.contains(key)); return _arguments[key]; }

protected:
    QHash<QString, QVariant> _arguments;

    friend class WorkSchedule;
};

class WorkSchedule
{
public:
    void setWorkingDir(const QString& path) { _working_dir = path; }
    const QString& workingDir() const { return _working_dir; }

    void addSpatialUpsample(int fine_level, int source_pass, int frame);
    void addRefine(int level, int last_coarse_pass, int this_pass, int time_step, int frame_number, int op_flags);
    void addInitialization(int level, int source_pass, int time_step, int frame_number, int op_flags);
    void addSynthesisInterpolated(int level, int pass, int frame, int time_step);
    void addCopyOutputToFinal(int first_frame, int last_frame, int final_pass);

    int numWorkUnits() const { return _work_units.size(); }
    const WorkUnit& workUnit(int index) const { return _work_units[index]; }

    bool save(const QString& filename);
    bool load(const QString& filename);
    bool spoolTinaJob();

protected:

    class TinaNode {
    public:
        ~TinaNode();

        QString _name;
        QList<TinaNode*> _full_subtasks;
        QList<TinaNode*> _instance_subtasks;
        QList<int> _units;
    };

    int makePassTree(int first, TinaNode* last_pass, TinaNode** out_node);
    QString commandsString(TinaNode* node);
    QString taskString(TinaNode* node);
    QString jobString(TinaNode* root);

protected:

    QList<WorkUnit> _work_units;
    QString _working_dir;
};

#endif
