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

#include <QtGui/QApplication>
#include <QtCore/QDebug>
#include <QtCore/QString>

#if _MSC_VER
#include <string.h>
#include "windows.h"
#endif

#include "qcommandline.h"
#include "nlConsoleInterface.h"
#include "nlMainWindow.h"

#include "PaintTweenCUDA/imageIO.h"

void msgHandler( QtMsgType type, const char* msg )
{
    Q_UNUSED(type)
    fprintf( stderr, "%s\n", msg );

#if _MSC_VER
    QString out = QString(msg) + QString("\n");
    wchar_t wcharstr[1024];
    int length = out.trimmed().left(1023).toWCharArray(wcharstr);
    wcharstr[length] = 0;
    OutputDebugStringW(wcharstr);
    OutputDebugStringA("\r\n");
#endif
}

int main(int argc, char *argv[])
{
    QCoreApplication::setOrganizationName("Pixar");
    QCoreApplication::setOrganizationDomain("pixar.com");
    QCoreApplication::setApplicationName("PaintTweenGUI");
    QCoreApplication::setApplicationVersion("0.2");


    QCommandLine cmdline(argc,argv);
    cmdline.addOption("file", "Path to working set .xml file", QCommandLine::Optional);
    cmdline.addOption("outDir", "Path to existing output directory", QCommandLine::Optional);
    cmdline.addOption("mode", "Default is 'gui'. For console mode, use 'still', 'anim', or 'batch'. For Nuke server mode, use 'nuke'.", QCommandLine::Optional);
    cmdline.addOption("schedule", "Work schedule file for mode=batch", QCommandLine::Optional);
    cmdline.addOption("tasks", "Task id list for mode=batch, comma separated", QCommandLine::Optional);
    cmdline.addOption("workingDir", "Path to working directory for temporary files", QCommandLine::Optional);
    cmdline.addOption("frame", "Frame to render in mode 'still' when using an image sequence (shot).",QCommandLine::Optional);
    cmdline.addOption("host", "Names of host to run on supplied by tina", QCommandLine::Optional);
    cmdline.addOption("device", "Directly specify the CUDA device to use", QCommandLine::Optional);
    cmdline.addOption("threads", "Number of threads", QCommandLine::Optional);
    cmdline.addSwitch("useCachedPreprocess", "Use cached keyframe preprocess images from a previous run", QCommandLine::Optional);
    cmdline.addSwitch("runLocal", "Run on the current machine, don't spool a job to tina", QCommandLine::Optional);
    cmdline.enableVersion(true); // enable -v // --version
    cmdline.enableHelp(true); // enable -h / --help

    cmdline.connect(&cmdline,SIGNAL(parseError(QString)),SLOT(printError(QString)));

    cmdline.parse();

    qInstallMsgHandler( msgHandler );

    QString mode = "";
    if(cmdline.optionsFound.contains("mode"))
        mode = cmdline.optionsFound["mode"].first();

    QString file = "";
    if(cmdline.optionsFound.contains("file"))
        file = cmdline.optionsFound["file"].first();

    QString schedule = "";
    if(cmdline.optionsFound.contains("schedule"))
        schedule = cmdline.optionsFound["schedule"].first();

    QString tasks = "";
    if(cmdline.optionsFound.contains("tasks"))
        tasks = cmdline.optionsFound["tasks"].first();

    QString outputDir = "";
    if(cmdline.optionsFound.contains("outDir"))
        outputDir = cmdline.optionsFound["outDir"].first();

    QString workingDir = "";
    if(cmdline.optionsFound.contains("workingDir"))
        workingDir = cmdline.optionsFound["workingDir"].first();

    QString host = "<host not specified>";
    if(cmdline.optionsFound.contains("host"))
        host = cmdline.optionsFound["host"].first();

    QString device = "";
    if(cmdline.optionsFound.contains("device"))
        device = cmdline.optionsFound["device"].first();

    int frame = 1;
    if(cmdline.optionsFound.contains("frame"))
        frame = cmdline.optionsFound["frame"].first().toInt();

    bool useCachedPreprocess = false;
    if(cmdline.switchsFound.contains("useCachedPreprocess"))
        useCachedPreprocess = cmdline.switchsFound["useCachedPreprocess"];

    bool runLocal = false;
    if(cmdline.switchsFound.contains("runLocal"))
        runLocal = cmdline.switchsFound["runLocal"];

    bool got_slot = false;
    int slot = host.section("_", -1, -1).toInt(&got_slot);
    int cudaDevice = 0;
    if (got_slot) {
        cudaDevice = slot - 1;
    } else {
        cudaDevice = device.toInt(&got_slot);
        if (!got_slot) {
            cudaDevice = 0;
        }
    }

    int numThreads = 0;
    if(cmdline.optionsFound.contains("threads"))
        numThreads = cmdline.optionsFound["threads"].first().toInt();
    ImageIO::multithread(numThreads);

    NLParameters::instance().setBool("use_cached_preprocess", useCachedPreprocess);
    NLParameters::instance().setBool("run_local", runLocal);


    // In case we have a console-only argument, we run the
    // console interface in the specified mode:
    if (mode == "still" || mode == "anim" || mode == "batch"){

        NLParameters::instance().setBool("run_offline", true);

        bool bad_input = false;
        int return_code = 0;
        if(file.isEmpty()) {
            qCritical() << "Empty working set file path! You must specify working set .xml file";
            bad_input = true;
        }
        if (outputDir.isEmpty()) {
            qCritical() << "Empty output directory path! You must specify an existing output directory";
            bad_input = true;
        }
        if (schedule.isEmpty() && mode == "batch") {
            qCritical() << "Empty work schedule path! You must specify a schedule for batch mode";
            bad_input = true;
        }

        if (!bad_input) {
            qDebug() << "PaintTween: running in console mode on " << host;

            NLConsoleInterface console(cudaDevice);


            if(console.setup(cmdline.arguments().at(0), file, outputDir, workingDir, schedule,
                             mode, tasks, frame)){
                if (!console.runAnimation()) {
                    return_code = -1;
                }
            } else {
                return_code = -1;
            }

            console.finalCleanUp();

        }

        return return_code;
    }

    NLParameters::instance().setBool("run_offline", false);

    QApplication app(argc, argv);
    // In case the 'mode' argument was not specified or 'gui' was chosen, we start the GUI:

    NLMainWindow* mainWindow = new NLMainWindow(cudaDevice);
    mainWindow->show();

    if (mode == "nuke") {
        mainWindow->startServer();
    }

    if (mode == "batchdebug") {
        mainWindow->setupBatchDebug(file, outputDir, workingDir, schedule, tasks);
    }
    else if(!file.isEmpty()){
        qDebug()<<"Reading working set file: "<< file;
        if(mainWindow->readWorkingSet(file))
            mainWindow->updateWindows();
    }

    app.connect(&app, SIGNAL(lastWindowClosed(void)), &app, SLOT(quit(void)));
    bool returnCode = app.exec();
    delete mainWindow;
    return returnCode;

    return 0;
}
