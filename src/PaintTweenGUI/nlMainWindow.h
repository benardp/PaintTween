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

#ifndef NLMAINWINDOW_H
#define NLMAINWINDOW_H

#include <QMainWindow>
#include <QComboBox>
#include <QAction>
#include <QVector>

#include "networkProtocol.h"
#include "nlPlayRangeWidget.h"
#include "nlSynthesizer.h"

class QTcpServer;
class NLSynthesisWidget;
class NLStyleTabWidget;
class NLImageViewerWidget;
class NLDialsAndKnobs;

class NLMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    NLMainWindow(int cudaDevice);
    ~NLMainWindow();

    bool readWorkingSet(const QString &fileName, bool styleRefreshOnly = false);
    void updateWindows();
    bool startServer();
    bool setupBatchDebug(const QString& fileName, const QString& outputDir,
                         const QString& workingDir, const QString& scheduleFile,
                         const QString& taskString);

public slots:
    // Slots for reading/writing working set:
    void saveWorkingSetToWorking();
    void onReadWorkingSet();
    void openRecentFile();
    void onSaveWorkingSet();
    void onSaveWorkingSetAs();

    void refreshInputViewers();
    void refreshStyleTabs();

    // Slot for skipping to a specific frame:
    void goToFrame(int frame);
    void goToLevel(int level);
    void goToPass(int pass);

    // Slot for updating/running synthesis:
    void updateSynthesisControls();
    void runAnimationSynthesis();

    // Client / server slots:
    void handleSocketConnection();
    void handleStartRequest();
    void handleStyleRefreshRequest();
    void handleFinishRequest();
    void setRequestOptions();
    void setupPostSynthesisHandling();

protected:
    void closeEvent(QCloseEvent* event);
    void createActions();
    void createToolBars();
    void createWidgets();
    void updateRecentFileActions();

    void createInputViewerWidgets();
    void createStyleWidgets();
    void createSynthesisViewerWidgets();
    void refreshSynthesisViewers();

    bool stopServer();

private:
    // Main control and data:
    NLSynthesizer* _synthesizer;

    QMenu* _viewMenu;
    QAction* _updateSynthesisAction;
    QAction* _pauseSynthesisAction;
    QAction* _stepSynthesisAction;
    QAction* _stepBackSynthesisAction;
    QAction* _resumeSynthesisAction;

    QVector<NLImageViewerWidget*> _inputViewers;
    QVector<NLSynthesisWidget*> _inputStyleViewers;
    QVector<NLSynthesisWidget*> _synthesisViewers;
    NLStyleTabWidget*_styleInputTabs;
    NLStyleTabWidget* _styleOutputTabs;
    NLDialsAndKnobs* _globalParametersWidget;

    QVector<QAction*> _recent_workingset_action;
    QString _most_recent_workingset;
    QString _animationSaveDir;

    NLPlayRangeWidget* _playRangeWidget;
    QComboBox* _colorspace;
    QComboBox* _background;

    // Client / server stuff
    QTcpServer* _tcpServer;
    NLNetworkMessage _currentNetworkRequest;
};

#endif // NLMAINWINDOW_H
