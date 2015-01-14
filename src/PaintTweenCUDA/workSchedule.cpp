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

#include "workSchedule.h"
#include "dataAccess.h"
#include "imageIO.h"

#include <QtCore/QFile>
#include <QtCore/QDebug>
#include <QtXml/QDomDocument>
#include <QtNetwork/QHostInfo>

WorkUnit::WorkUnit(int level, int pass, int frame)
{
    _arguments["level"] = level;
    _arguments["pass"] = pass;
    _arguments["frame"] = frame;
}

WorkUnit::WorkUnit(const QDomElement& elem)
{
    QDomNamedNodeMap attrs = elem.attributes();
    for (int i = 0; i < attrs.size(); i++) {
        QDomAttr attr = attrs.item(i).toAttr();
        _arguments[attr.name()] = attr.value();
    }
}

void WorkUnit::save(QDomElement& elem)
{
    QList<QString> keys = _arguments.keys();
    foreach(QString key, keys) {
        elem.setAttribute(key, _arguments[key].toString());
    }
}

void WorkSchedule::addSpatialUpsample(int fine_level, int source_pass, int frame)
{
    WorkUnit task(fine_level, 0, frame);
    task._arguments["type"] = "spatial_upsample";
    task._arguments["coarse_level"] = fine_level + 1;
    task._arguments["source_pass"] = source_pass;
    _work_units.append(task);
}

void WorkSchedule::addRefine(int level, int last_coarse_pass, int this_pass, int time_step, int frame_number, int op_flags)
{
    WorkUnit task(level, this_pass, frame_number);
    task._arguments["type"] = "refine";
    task._arguments["last_coarse_pass"] = last_coarse_pass;
    task._arguments["time_step"] = time_step;
    task._arguments["op_flags"] = op_flags;
    _work_units.append(task);
}

void WorkSchedule::addInitialization(int level, int source_pass, int time_step, int frame_number, int op_flags)
{
    WorkUnit task(level, 0, frame_number);
    task._arguments["type"] = "initialize";
    task._arguments["time_step"] = time_step;
    task._arguments["op_flags"] = op_flags;
    task._arguments["source_pass"] = source_pass;
    _work_units.append(task);
}

void WorkSchedule::addSynthesisInterpolated(int level, int pass, int frame, int time_step)
{
    WorkUnit task(level, pass, frame);
    task._arguments["type"] = "synthesis_interpolated";
    task._arguments["time_step"] = time_step;
    _work_units.append(task);
}

void WorkSchedule::addCopyOutputToFinal(int first_frame, int last_frame, int final_pass)
{
    WorkUnit task(0, final_pass+1, first_frame);
    task._arguments["type"] = "copy_output_to_final";
    task._arguments["first_frame"] = first_frame;
    task._arguments["last_frame"] = last_frame;
    task._arguments["final_pass"] = final_pass;
    _work_units.append(task);
}

bool WorkSchedule::save(const QString& filename)
{
    QDomDocument doc("workschedule");
    QDomElement root = doc.createElement("workschedule");
    root.setAttribute("working_dir", _working_dir);
    doc.appendChild(root);

    for (int i = 0; i < _work_units.size(); i++) {
        QDomElement elem = doc.createElement("task");
        _work_units[i].save(elem);
        elem.setAttribute("id", i);
        root.appendChild(elem);
    }

    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)) {
        qCritical("Could not write work schedule %s", qPrintable(filename));
        return false;
    }

    qDebug("Writing work schedule to %s", qPrintable(filename));

    file.write(doc.toByteArray());

    return true;
}

bool WorkSchedule::load(const QString& filename)
{
    QDomDocument wsDoc("workschedule");

    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly))
    {
        qCritical("Could not open schedule %s", qPrintable(filename));
        return false;
    }
    if (!wsDoc.setContent(&file)) {
        file.close();
        qCritical("Could not parse schedule %s", qPrintable(filename));
        return false;
    }
    file.close();

    QDomElement root = wsDoc.documentElement();
    if(root.isNull()) {
        qWarning() << "Empty schedule";
        return false;
    }

    _working_dir = root.attribute("working_dir");

    QDomElement elem = root.firstChildElement("task");
    while (!elem.isNull()) {
        WorkUnit t(elem);
        _work_units.push_back(t);
        elem = elem.nextSiblingElement();
    }

    return true;
}

WorkSchedule::TinaNode::~TinaNode()
{
    qDeleteAll(_full_subtasks);
}

int WorkSchedule::makePassTree(int first, TinaNode* last_pass, TinaNode** out_node)
{
    int level = _work_units[first].level();
    int pass = _work_units[first].pass();

    TinaNode* pass_node = new TinaNode;
    pass_node->_name = QString("level_%1_pass_%2").arg(level).arg(pass);

    TinaNode* unit_node = new TinaNode;
    if (_work_units[first].type() == "copy_output_to_final"){
        unit_node->_name = QString("copy_output_to_final");
    } else {
        unit_node->_name = QString("%4_l%1p%2f%3").arg(level).arg(pass).arg(_work_units[first].frame()).arg(_work_units[first].type());
    }
    if (last_pass) {
        unit_node->_full_subtasks.push_back(last_pass);
    }
    unit_node->_units.push_back(first);

    pass_node->_full_subtasks.push_back(unit_node);

    int i = first + 1;
    while (i < _work_units.size() && _work_units[i].level() == level && _work_units[i].pass() == pass) {
        unit_node = new TinaNode;
        unit_node->_name = QString("%4_l%1p%2f%3").arg(level).arg(pass).arg(_work_units[i].frame()).arg(_work_units[i].type());
        if (last_pass) {
            unit_node->_instance_subtasks.push_back(last_pass);
        }
        unit_node->_units.push_back(i);

        pass_node->_full_subtasks.push_back(unit_node);
        i++;
    }
    *out_node = pass_node;

    return i;
}

QString job_template("Job -title {%1} -memory 2G -memlimit 100G -pbias 100 -crews {%2} -subtasks {%3}\n");
QString cmd_template("RemoteCmd {/bin/tcsh -c {%1 -mode batch -host \"%s\" -file %2 -outDir %3 -workingDir %4 -schedule %5 -tasks %6}}\n");
QString task_template("Task {%1} -memory 2G -memlimit 100G -service {linux64,RENDER} -cmds {%2}\n");
QString task_subtask_template("Task {%1} -memory 2G -memlimit 100G -service {linux64,RENDER} -subtasks {\n%2} -cmds {%3}\n");
QString instance_template("Instance {%1}\n");



QString WorkSchedule::commandsString(TinaNode* node)
{
    if (node->_units.isEmpty()) {
        return QString();
    }

    const DataAccess& data = DataAccess::instance();
    QString exec = data.getStringParameter("app_path");
    // Hack to change name to "PaintTweenGUI", not "PaintTweenGUI2" for the tina hack to work.
    if (exec.endsWith("PaintTweenGUI2")) {
        exec = exec.left(exec.size()-1);
    }
    QString file = data.getStringParameter("working_set_path");
    QString outDir = data.getOutDir();
    QString workingDir = ImageIO::netAddressablePath(QHostInfo::localHostName(), data.getTemporaryDir());
    QString scheduleFile = QString("%1/schedule.xml").arg(workingDir);
    QString tasks;
    for (int i = 0; i < node->_units.size(); i++) {
        tasks.append(QString("%1,").arg(node->_units[i]));
    }
    QString cmd = cmd_template.arg(exec).arg(file).arg(outDir).arg(workingDir).arg(scheduleFile).arg(tasks);
    return cmd;
}

QString WorkSchedule::taskString(TinaNode* node)
{
    QString subtasks;
    for (int i = 0; i < node->_full_subtasks.size(); i++) {
        subtasks += taskString(node->_full_subtasks[i]);
    }
    for (int i = 0; i < node->_instance_subtasks.size(); i++) {
        subtasks += instance_template.arg(node->_instance_subtasks[i]->_name);
    }

    QString task;
    QString cmd = commandsString(node);
    if (subtasks.isEmpty()) {
        task = task_template.arg(node->_name).arg(cmd);
    } else {
        task = task_subtask_template.arg(node->_name).arg(subtasks).arg(cmd);
    }
    return task;
}

QString WorkSchedule::jobString(TinaNode* root)
{
    const DataAccess& data = DataAccess::instance();
    QString task = taskString(root);
    QString render_crews = data.getStringParameter("render_crews");
    QString path = data.getStringParameter("working_set_path");
    QString job = job_template.arg(path).arg(render_crews).arg(task);
    return job;
}

bool WorkSchedule::spoolTinaJob()
{
    const DataAccess& data = DataAccess::instance();

    QString schedule_file = QString("%1/schedule.xml").arg(data.getTemporaryDir());
    save(schedule_file);

    TinaNode* root;
    int next = makePassTree(0, 0, &root);
    while (next < _work_units.size()) {
        TinaNode* last = root;
        next = makePassTree(next, last, &root);
    }

    QString job_str = jobString(root);
    delete root;

    QString tinafile = QString("%1/spool.alf").arg(data.getTemporaryDir());

    QFile file(tinafile);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qCritical("Could not write %s", qPrintable(tinafile));
        return false;
    }

    file.write(job_str.toUtf8());
    file.close();

    QString creator = data.getStringParameter("working_set_creator");
    QString cmd = QString("tinaspool -h %2 --nomail --nosave %1").arg(tinafile).arg(creator);
    int ret = system(cmd.toUtf8());
    return ret == 0;
}
