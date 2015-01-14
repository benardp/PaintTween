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

#include "nlMainWindow.h"

#include <QtNetwork/QTcpServer>
#include <QtNetwork/QTcpSocket>
#include <QtCore/QTimer>
#include <QtCore/QDir>
#include <QtCore/QSettings>
#include <QtGui/QCloseEvent>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QToolBar>
#include <QtGui/QLabel>
#include <QtGui/QStatusBar>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QDockWidget>
#include <QtGui/QVBoxLayout>
#include <QtGui/QScrollArea>

#include "nlAnimationDialog.h"
#include "nlStyleTabWidget.h"
#include "nlSynthesisWidget.h"
#include "nlDialsAndKnobs.h"

const int MAX_RECENT_WORKINGSET = 9;

NLMainWindow::NLMainWindow(int cudaDevice)
    : QMainWindow(NULL),
      _synthesizer(NULL),
      _tcpServer(NULL)
{
    setWindowTitle("PaintTween");

    _synthesizer = &NLSynthesizer::instance();
    _synthesizer->initialize(this,cudaDevice);
    connect(_synthesizer, SIGNAL(animationDone()), this, SLOT(updateSynthesisControls()));
    connect(_synthesizer, SIGNAL(synthesisStarted()), this, SLOT(updateSynthesisControls()));
    connect(NLParameters::instancePtr(), SIGNAL(parametersChanged()), _synthesizer, SLOT(updateParameters()));
    _animationSaveDir = "";

    createActions();
    createToolBars();
    createWidgets();

    // Initialize the application status bar:
    statusBar()->showMessage(tr("Ready"));
}

NLMainWindow::~NLMainWindow()
{
}

void NLMainWindow::closeEvent(QCloseEvent* event)
{
    event->accept();

    // Closing the main window means closing the application. Make sure, that
    // all CUDA cleanup is properly performed. Unfortunately this code cannot
    // be put into destructors, which would yield a driver error.
    _synthesizer->finalCleanUp();
}

void NLMainWindow::createActions()
{
    // Create working set loading/saving actions in the menu:
    QAction* readWorkingSetAction = new QAction(tr("&Read Working Set..."), this);
    readWorkingSetAction->setShortcut(QKeySequence("Ctrl+O"));
    connect(readWorkingSetAction, SIGNAL(triggered(void)), this, SLOT(onReadWorkingSet(void)));

    _recent_workingset_action.resize(MAX_RECENT_WORKINGSET);
    for (int i = 0; i < MAX_RECENT_WORKINGSET; ++i) {
        _recent_workingset_action[i] = new QAction(this);
        _recent_workingset_action[i]->setVisible(false);
        connect(_recent_workingset_action[i], SIGNAL(triggered()),
                this, SLOT(openRecentFile()));
    }

    QAction*  saveWorkingSetAction = new QAction(tr("&Save Working Set..."), this);
    saveWorkingSetAction->setShortcut(QKeySequence("Ctrl+S"));
    connect(saveWorkingSetAction, SIGNAL(triggered(void)), this, SLOT(onSaveWorkingSet(void)));

    QAction* saveWorkingSetAsAction = new QAction(tr("&Save Working Set As..."), this);
    saveWorkingSetAsAction->setShortcut(QKeySequence("Ctrl+Shift+S"));
    connect(saveWorkingSetAsAction, SIGNAL(triggered(void)), this, SLOT(onSaveWorkingSetAs(void)));

    _updateSynthesisAction = new QAction(QIcon(":/icons/Refresh.png"), tr("&Update Synthesis..."), this);
    _updateSynthesisAction->setToolTip(tr("Update/Run Synthesis..."));
    connect(_updateSynthesisAction, SIGNAL(triggered(void)), _synthesizer, SLOT(updateSynthesis(void)));

    _pauseSynthesisAction = new QAction(QIcon(":/icons/Pause.png"), tr("&Pause Synthesis"), this);
    _pauseSynthesisAction->setToolTip(tr("Pause Synthesis"));
    connect(_pauseSynthesisAction, SIGNAL(triggered(void)), _synthesizer, SLOT(pauseSynthesis(void)));
    _pauseSynthesisAction->setEnabled(false);

    _resumeSynthesisAction = new QAction(QIcon(":/icons/Forward.png"), tr("&Resume Synthesis"), this);
    _resumeSynthesisAction->setToolTip(tr("Resume Synthesis"));
    connect(_resumeSynthesisAction, SIGNAL(triggered(void)), _synthesizer, SLOT(resumeSynthesis(void)));
    _resumeSynthesisAction->setEnabled(false);

    _stepSynthesisAction = new QAction(QIcon(":/icons/Step.png"), tr("Step &Forward"), this);
    _stepSynthesisAction->setToolTip(tr("Step Forward"));
    connect(_stepSynthesisAction, SIGNAL(triggered(void)), _synthesizer, SLOT(singleStepSynthesis(void)));

    _stepBackSynthesisAction = new QAction(QIcon(":/icons/StepBack.png"), tr("Step &Back"), this);
    _stepBackSynthesisAction->setToolTip(tr("Step Back"));
    connect(_stepBackSynthesisAction, SIGNAL(triggered(void)), _synthesizer, SLOT(singleStepBackSynthesis(void)));

    // Create the main window menu and add actions:
    QMenu* fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(readWorkingSetAction);
    fileMenu->addAction(saveWorkingSetAction);
    fileMenu->addAction(saveWorkingSetAsAction);
    QMenu* recentMenu = fileMenu->addMenu(tr("Recent..."));
    for (int i = 0; i < MAX_RECENT_WORKINGSET; ++i){
        _recent_workingset_action[i]->setShortcut(QKeySequence(tr((QString("Alt+Shift+%1").arg(i+1)).toStdString().c_str())));
        recentMenu->addAction(_recent_workingset_action[i]);
    }
    updateRecentFileActions();

    QAction* animationSynthesisAction = new QAction(tr("&Run Animation Synthesis..."), this);
    connect(animationSynthesisAction, SIGNAL(triggered(void)), this, SLOT(runAnimationSynthesis(void)));

    QMenu* toolsMenu = menuBar()->addMenu(tr("&Tools"));
    toolsMenu->addAction(animationSynthesisAction);

    QAction* examineCudaAction = new QAction(tr("&Examine Cuda"), this);
    connect(examineCudaAction, SIGNAL(triggered()), _synthesizer, SLOT(examineCuda()));
    toolsMenu->addAction(examineCudaAction);

    // View menu:
    _viewMenu = menuBar()->addMenu(tr("&View"));
}

void NLMainWindow::updateRecentFileActions()
{
    QSettings settings("PaintTweenGUI", "PaintTweenGUI");
    QStringList files = settings.value("recentFileList").toStringList();

    int numRecentFiles = qMin(files.size(), (int)MAX_RECENT_WORKINGSET);

    for (int i = 0; i < numRecentFiles; ++i) {
        QString text = tr("&%1 %2").arg(i + 1).arg(files[i]);
        _recent_workingset_action[i]->setText(text);
        _recent_workingset_action[i]->setData(files[i]);
        _recent_workingset_action[i]->setVisible(true);
        _recent_workingset_action[i]->setShortcut(QKeySequence(tr(((QString("Alt+Shift+%1").arg(i+1))).toStdString().c_str())));
    }
    for (int j = numRecentFiles; j < MAX_RECENT_WORKINGSET; ++j)
        _recent_workingset_action[j]->setVisible(false);
}

void NLMainWindow::createToolBars()
{
    setUnifiedTitleAndToolBarOnMac(true);
    setIconSize(QSize(25,25));

    // Create the main toolbar:
    QToolBar* toolBar = new QToolBar(this);
    toolBar->addAction(_updateSynthesisAction);
    toolBar->addAction(_pauseSynthesisAction);
    toolBar->addAction(_stepBackSynthesisAction);
    toolBar->addAction(_stepSynthesisAction);
    toolBar->addAction(_resumeSynthesisAction);
    toolBar->addSeparator();

    // Create play range widget (for controlling start and end frame, etc.):
    _playRangeWidget = new NLPlayRangeWidget(toolBar, _synthesizer);
    toolBar->addWidget(_playRangeWidget);
    connect(_playRangeWidget, SIGNAL(startFrameChanged(int)), _synthesizer, SLOT(updateFirstFrame(int)));
    connect(_playRangeWidget, SIGNAL(endFrameChanged(int)), _synthesizer, SLOT(updateLastFrame(int)));
    connect(_playRangeWidget, SIGNAL(curFrameChanged(int)), this, SLOT(goToFrame(int)));
    connect(_playRangeWidget, SIGNAL(synthesisModeChanged(int)), _synthesizer, SLOT(updateRealtimeSynthesisMode(int)));
    connect(_playRangeWidget, SIGNAL(curLevelChanged(int)), this, SLOT(goToLevel(int)));
    connect(_playRangeWidget, SIGNAL(curPassChanged(int)), this, SLOT(goToPass(int)));
    connect(_synthesizer, SIGNAL(synthesisAdvanced()), _playRangeWidget, SLOT(update()));

    toolBar->addSeparator();
    _colorspace = new QComboBox;
    _colorspace->addItem("linear");
    _colorspace->addItem("sRGB");
    toolBar->addWidget(new QLabel(" Color space: "));
    toolBar->addWidget(_colorspace);

    _background = new QComboBox;
    _background->addItem("checkerboard");
    _background->addItem("black");
    toolBar->addWidget(new QLabel(" Background: "));
    toolBar->addWidget(_background);

    addToolBar(toolBar);
}

void NLMainWindow::createWidgets()
{
    createInputViewerWidgets();
    createStyleWidgets();
    createSynthesisViewerWidgets();

    QDockWidget* globalParametersDock = new QDockWidget(tr("Global Parameters"), this);
    _globalParametersWidget = new NLDialsAndKnobs(globalParametersDock);
    _globalParametersWidget->setSizePolicy(QSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum));
    QScrollArea* scrollArea = new QScrollArea;
    scrollArea->setWidget(_globalParametersWidget);
    globalParametersDock->setAllowedAreas(Qt::BottomDockWidgetArea);
    globalParametersDock->setWidget(scrollArea);
    addDockWidget(Qt::BottomDockWidgetArea, globalParametersDock);
    _viewMenu->addAction(globalParametersDock->toggleViewAction());

    connect(_styleOutputTabs, SIGNAL(currentStyleChanged(int)), &NLParameters::instance(), SLOT(styleChanged(int)));
}

static QVector<InputElements> kInputViewerWidgetEnums = QVector<InputElements>() << IN_INPUT << IN_ORIENTATION << IN_DIST_TRANS << IN_SCALE;
static QStringList kInputViewerWidgetNames = QStringList() << "Image Input" << "Image Orientation" << "Image Dist. Trans." << "Image Scale";

void NLMainWindow::createInputViewerWidgets()
{
    QVector<QDockWidget*> docks;
    QDockWidget* dock;
    for (int i = 0; i < kInputViewerWidgetEnums.size(); i++) {
        dock = new QDockWidget(kInputViewerWidgetNames.at(i), this);
        NLImageViewerWidget* inputWidget = new NLImageViewerWidget(dock,_synthesizer,kInputViewerWidgetEnums.at(i));
        QSizePolicy sizePolicy;
        sizePolicy.setHorizontalPolicy(QSizePolicy::Preferred);
        sizePolicy.setVerticalPolicy(QSizePolicy::Preferred);
        inputWidget->setSizePolicy(sizePolicy);
        dock->setAllowedAreas(Qt::TopDockWidgetArea);
        dock->setWidget(inputWidget);
        addDockWidget(Qt::TopDockWidgetArea, dock);
        _inputViewers.append(inputWidget);
        docks.append(dock);
        if(kInputViewerWidgetEnums.at(i) == IN_INPUT){
            connect(_background, SIGNAL(currentIndexChanged(int)), inputWidget, SLOT(changeBackground(int)));
            connect(_colorspace, SIGNAL(currentIndexChanged(int)), inputWidget, SLOT(changeColorSpace(int)));
        }
        connect(_synthesizer, SIGNAL(parameterChanged()), inputWidget, SLOT(update()));
    }

    for (int i = 1; i < docks.size(); i++) {
        tabifyDockWidget(docks[i-1], docks[i]);
    }
    docks.first()->show();
    docks.first()->raise();
}

static QVector<StyleElements> kStyleViewerWidgetEnums = QVector<StyleElements>() << STYLE_INPUT<< STYLE_ORIENTATION << STYLE_DIST_TRANS << STYLE_OUTPUT;
static QStringList kStyleViewerWidgetNames = QStringList() << "Style Input" << "Style Orientation" << "Style Dist. Trans." << "Style Output";

void NLMainWindow::createStyleWidgets()
{
    // Create viewer widget for the input style(s):
    QDockWidget* inputStyleDock = new QDockWidget(tr("Style Input"), this);
    _styleInputTabs = new NLStyleTabWidget(inputStyleDock);
    _styleInputTabs->setTabPosition(QTabWidget::South);
    QSizePolicy sizePolicy;
    sizePolicy.setHorizontalPolicy(QSizePolicy::Preferred);
    sizePolicy.setVerticalPolicy(QSizePolicy::Preferred);
    _styleInputTabs->setSizePolicy(sizePolicy);
    inputStyleDock->setAllowedAreas(Qt::TopDockWidgetArea);
    inputStyleDock->setWidget(_styleInputTabs);
    addDockWidget(Qt::TopDockWidgetArea, inputStyleDock);
    _viewMenu->addAction(inputStyleDock->toggleViewAction());

    // Create viewer widget for the output style(s):
    QDockWidget* outputStyleDock = new QDockWidget(tr("Style Output"), this);
    _styleOutputTabs = new NLStyleTabWidget();
    _styleOutputTabs->setTabPosition(QTabWidget::South);
    _styleOutputTabs->setSizePolicy(sizePolicy);
    outputStyleDock->setAllowedAreas(Qt::TopDockWidgetArea);
    outputStyleDock->setWidget(_styleOutputTabs);
    addDockWidget(Qt::TopDockWidgetArea, outputStyleDock);
    _viewMenu->addAction(outputStyleDock->toggleViewAction());

    connect(_styleInputTabs, SIGNAL(currentStyleChanged(int)),_styleOutputTabs, SLOT(setCurrentStyle(int)));
    connect(_styleOutputTabs, SIGNAL(currentStyleChanged(int)),_styleInputTabs, SLOT(setCurrentStyle(int)));

    for(int j=0; j<kStyleViewerWidgetEnums.size(); j++){
        NLStyleViewerWidget* thisStyle;
        if(kStyleViewerWidgetEnums.at(j) == STYLE_OUTPUT){
            thisStyle = new NLStyleViewerWidget(_styleOutputTabs,_synthesizer,0,kStyleViewerWidgetEnums.at(j));
            _styleOutputTabs->addTab(thisStyle, kStyleViewerWidgetNames.at(j));
        }else{
            thisStyle = new NLStyleViewerWidget(_styleInputTabs,_synthesizer,0,kStyleViewerWidgetEnums.at(j));
            _styleInputTabs->addTab(thisStyle, kStyleViewerWidgetNames.at(j));
        }
    }
}

static QVector<TsOutputType> kOutputViewerWidgetEnums =
        QVector<TsOutputType>() << TS_OUTPUT_CANVAS << TS_OUTPUT_OFFSET
                                << TS_OUTPUT_RESIDUAL << TS_OUTPUT_VEL_F << TS_OUTPUT_VEL_B
                                << TS_OUTPUT_DIST_TRANS << TS_OUTPUT_ADVECTED_F << TS_OUTPUT_ADVECTED_B
                                << TS_OUTPUT_RIBBON_F << TS_OUTPUT_RIBBON_B << TS_OUTPUT_ID;
static QStringList kOutputViewerWidgetNames =
        QStringList() << "Synthesis" << "Offsets" << "Residual"
                      << "Vel. F." << "Vel. B." << "Dist. Trans." << "Adv. F."<< "Adv. B."
                      << "Ribbon F" << "Ribbon B" << "Id";

void NLMainWindow::createSynthesisViewerWidgets()
{
    QSizePolicy sizePolicy;
    sizePolicy.setHorizontalPolicy(QSizePolicy::MinimumExpanding);
    sizePolicy.setVerticalPolicy(QSizePolicy::MinimumExpanding);
    QVector<QDockWidget*> docks;
    QDockWidget* dock;
    for (int i = 0; i < kOutputViewerWidgetEnums.size(); i++) {
        dock = new QDockWidget(kOutputViewerWidgetNames.at(i), this);
        NLSynthesisWidget* syn_widget = new NLSynthesisWidget(dock, _synthesizer, kOutputViewerWidgetEnums[i]);
        syn_widget->setSizePolicy(sizePolicy);
        QWidget* holder = new QWidget;
        QVBoxLayout* layout = new QVBoxLayout;
        layout->addWidget(syn_widget);
        dock->setAllowedAreas(Qt::BottomDockWidgetArea);
        if (syn_widget->controlWidget()) {
            layout->addWidget(syn_widget->controlWidget());
            layout->setStretch(0,1);
        }
        holder->setLayout(layout);
        dock->setWidget(holder);
        addDockWidget(Qt::BottomDockWidgetArea, dock);
        _viewMenu->addAction(dock->toggleViewAction());

        connect(syn_widget, SIGNAL(imagePointRefreshed(QPoint, float, float, int)),
                _styleOutputTabs, SLOT(drawBox(QPoint, float, float, int)));
        connect(syn_widget, SIGNAL(imagePointRefreshed(QPoint, float, float, int)),
                _styleInputTabs, SLOT(drawBox(QPoint, float, float, int)));
        connect(syn_widget, SIGNAL(imagePointClicked(int)),
                _styleOutputTabs, SLOT(updateIndexFromStyle(int)));
        connect(syn_widget, SIGNAL(imagePointClicked(int)),
                _styleInputTabs, SLOT(updateIndexFromStyle(int)));
        connect(_synthesizer, SIGNAL(parameterChanged()), syn_widget, SLOT(update()));

        connect(dock, SIGNAL(visibilityChanged(bool)), syn_widget, SLOT(setVisible(bool)));
        connect(_synthesizer, SIGNAL(synthesisStarted()), syn_widget, SLOT(removeVizBox()));

        if(kOutputViewerWidgetEnums.at(i) == TS_OUTPUT_CANVAS ||
                kOutputViewerWidgetEnums.at(i) == TS_OUTPUT_ADVECTED_F ||
                kOutputViewerWidgetEnums.at(i) == TS_OUTPUT_ADVECTED_B){
            connect(_colorspace, SIGNAL(currentIndexChanged(int)), syn_widget, SLOT(changeColorSpace(int)));
            connect(_background, SIGNAL(currentIndexChanged(int)), syn_widget, SLOT(changeBackground(int)));
            syn_widget->changeBackground(0);
        }

        _synthesisViewers.push_back(syn_widget);
        docks.push_back(dock);
    }

    // Create offsets histogram viewer widget and dock:
    dock = new QDockWidget(tr("Histogram"), this);
    NLHistogramWidget* histogramViewerWidget = new NLHistogramWidget(dock, _synthesizer);
    histogramViewerWidget->setSizePolicy(sizePolicy);
    dock->setAllowedAreas(Qt::BottomDockWidgetArea);
    QWidget* holder = new QWidget;
    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(histogramViewerWidget);
    holder->setLayout(layout);
    dock->setWidget(holder);
    addDockWidget(Qt::BottomDockWidgetArea, dock);
    _viewMenu->addAction(dock->toggleViewAction());

    connect(histogramViewerWidget, SIGNAL(imagePointRefreshed(QPoint, float, float, int)),
            _styleOutputTabs, SLOT(drawBox(QPoint, float, float, int)));
    connect(histogramViewerWidget, SIGNAL(imagePointRefreshed(QPoint, float, float, int)),
            _styleInputTabs, SLOT(drawBox(QPoint, float, float, int)));
    connect(histogramViewerWidget, SIGNAL(imagePointClicked(int)),
            _styleOutputTabs, SLOT(updateIndexFromStyle(int)));
    connect(histogramViewerWidget, SIGNAL(imagePointClicked(int)),
            _styleInputTabs, SLOT(updateIndexFromStyle(int)));
    connect(_styleOutputTabs, SIGNAL(currentStyleChanged(int)),histogramViewerWidget, SLOT(styleChanged(int)));

    connect(dock, SIGNAL(visibilityChanged(bool)), histogramViewerWidget, SLOT(setVisible(bool)));
    connect(_synthesizer, SIGNAL(synthesisStarted()), histogramViewerWidget, SLOT(removeVizBox()));
    connect(histogramViewerWidget, SIGNAL(statusMessage(QString)), statusBar(), SLOT(showMessage(QString)));

    _synthesisViewers.push_back(histogramViewerWidget);
    docks.push_back(dock);

    for (int i = docks.size() - 1; i >= 1; i--) {
        tabifyDockWidget(docks[i], docks[i-1]);
    }
}

void NLMainWindow::updateWindows()
{
    refreshInputViewers();
    refreshStyleTabs();

    _animationSaveDir = "";
    disconnect(this, SLOT(saveWorkingSetToWorking()));

    _playRangeWidget->updateRangeFromSynthesizer();
}


bool NLMainWindow::readWorkingSet(const QString& fileName, bool styleRefreshOnly)
{
    _most_recent_workingset = fileName;
    return _synthesizer->readWorkingSet(fileName, styleRefreshOnly);
}

void NLMainWindow::saveWorkingSetToWorking()
{
    // Save the working set to an appropriately named file in the working directory. Usually called when serving Nuke.
    if (_synthesizer && !_animationSaveDir.isEmpty()) {
        QString working_dir = _synthesizer->getOutDir();

        QString working_filename = QDir::cleanPath(working_dir + "working.xml");
        _synthesizer->writeWorkingSet(working_filename);
    }
}

void NLMainWindow::updateSynthesisControls()
{
    if (_synthesizer) {
        if (_synthesizer->isRealtimeSynthesisPaused()) {
            _pauseSynthesisAction->setEnabled(false);
            _resumeSynthesisAction->setEnabled(true);
            _stepSynthesisAction->setEnabled(true);
            _stepBackSynthesisAction->setEnabled(true);
        } else if (_synthesizer->isRealtimeSynthesisRunning()) {
            _pauseSynthesisAction->setEnabled(true);
            _resumeSynthesisAction->setEnabled(false);
            _stepSynthesisAction->setEnabled(false);
            _stepBackSynthesisAction->setEnabled(false);
        } else {
            _pauseSynthesisAction->setEnabled(false);
            _resumeSynthesisAction->setEnabled(false);
            _stepSynthesisAction->setEnabled(true);
            _stepBackSynthesisAction->setEnabled(false);
        }
    }
}

void NLMainWindow::refreshInputViewers()
{
    for (int i = 0; i < kInputViewerWidgetEnums.size(); i++){
        _inputViewers.at(i)->update();
    }
}

void NLMainWindow::refreshStyleTabs()
{
    _styleInputTabs->clear();
    _styleOutputTabs->clear();

    int numStyles = _synthesizer->getNumStyles();

    for(int i = 0; i < numStyles; i++){
        QString tabNum;
        tabNum.setNum(i);

        for(int j=0; j<kStyleViewerWidgetEnums.size(); j++){
            NLStyleViewerWidget* thisStyle;
            if(kStyleViewerWidgetEnums.at(j) == STYLE_OUTPUT){
                thisStyle = new NLStyleViewerWidget(_styleOutputTabs,_synthesizer,i,kStyleViewerWidgetEnums.at(j));
                _styleOutputTabs->addTab(thisStyle, kStyleViewerWidgetNames.at(j) +" "+ tabNum);
            }else{
                thisStyle = new NLStyleViewerWidget(_styleInputTabs,_synthesizer,i,kStyleViewerWidgetEnums.at(j));
                _styleInputTabs->addTab(thisStyle, kStyleViewerWidgetNames.at(j) +" "+ tabNum);
            }
            bool isColorImage = (kStyleViewerWidgetEnums.at(j) == STYLE_INPUT || kStyleViewerWidgetEnums.at(j) == STYLE_OUTPUT);
            if(isColorImage){
                connect(_background, SIGNAL(currentIndexChanged(int)), thisStyle, SLOT(changeBackground(int)));
                connect(_colorspace, SIGNAL(currentIndexChanged(int)), thisStyle, SLOT(changeColorSpace(int)));
            }
        }
    }

    _styleInputTabs->setNumTabsPerStyle(kStyleViewerWidgetEnums.size()-1);
    _styleOutputTabs->setNumTabsPerStyle(1);

    // Skip the synthesis viewer widget (i = 1).
    for (int i = 1; i < _synthesisViewers.size(); i++) {
        _synthesisViewers[i]->setVisible(false);
    }
}

void NLMainWindow::refreshSynthesisViewers()
{
    for (int i = 0; i < _synthesisViewers.size(); i++) {
        _synthesisViewers.at(i)->update();
    }
}

void NLMainWindow::goToFrame(int frame)
{
    if(_synthesizer->updateCurrentFrame(frame)){
        refreshInputViewers();
        refreshSynthesisViewers();
        updateSynthesisControls();
    } else
        qDebug()<<"PaintTween: DANGER! Error in MainWindow::goToFrame(-) attempting to access frame "<<frame;
}

void NLMainWindow::goToLevel(int level)
{
    if(_synthesizer->updateViewerLevel(level)){
        refreshSynthesisViewers();
        updateSynthesisControls();
    } else
        qDebug()<<"PaintTween: DANGER! Error in MainWindow::goToLevel(-) attempting to access level "<<level;
}

void NLMainWindow::goToPass(int pass)
{
    if(_synthesizer->updateViewerPass(pass)){
        refreshSynthesisViewers();
        updateSynthesisControls();
    } else
        qDebug()<<"PaintTween: DANGER! Error in MainWindow::goToPass(-) attempting to access pass "<<pass;
}

void NLMainWindow::runAnimationSynthesis()
{
    NLAnimationDialog* animationDialog = new NLAnimationDialog(NULL, _synthesizer, _animationSaveDir);
    animationDialog->exec();
    delete animationDialog;
}

void NLMainWindow::onReadWorkingSet()
{
    // Function for reading a working set (xml file):
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Working Set File"), QDir::currentPath());
    if (!fileName.isEmpty()){
        if(!readWorkingSet(fileName)){
            QMessageBox::information(this, tr("Working Set"), tr("Cannot load working set %1.").arg(fileName));
        }else{
            QSettings settings("PaintTweenGUI", "PaintTweenGUI");
            QStringList files = settings.value("recentFileList").toStringList();
            files.removeAll(fileName);
            files.prepend(fileName);
            while (files.size() > MAX_RECENT_WORKINGSET)
                files.removeLast();
            settings.setValue("recentFileList", files);
            updateRecentFileActions();

            updateWindows();
        }
    }
}

void NLMainWindow::openRecentFile()
{
    QAction *action = qobject_cast<QAction *>(sender());
    if (action){
        if(!readWorkingSet(action->data().toString())){
            QMessageBox::information(this, tr("Working Set"), tr("Cannot load working set %1.").arg(action->data().toString()));
        }else{
            updateWindows();
        }
    }
}

void NLMainWindow::onSaveWorkingSet()
{
    if (!_most_recent_workingset.isEmpty()){
        if (!_synthesizer->writeWorkingSet(_most_recent_workingset)){
            QMessageBox::information(this, tr("Working Set"), tr("Cannot save working set %1.").arg(_most_recent_workingset));
        }
    }
}

void NLMainWindow::onSaveWorkingSetAs()
{
    // Function for saving a working set (xml file):
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save Working Set File"), QDir::currentPath());
    if (!fileName.isEmpty()){
        if (!_synthesizer->writeWorkingSet(fileName)){
            QMessageBox::information(this, tr("Working Set"), tr("Cannot save working set %1.").arg(fileName));
        }
    }
}

bool NLMainWindow::setupBatchDebug(const QString& fileName, const QString& outputDir,
                               const QString& workingDir, const QString& scheduleFile,
                               const QString& taskString)
{
    bool success = _synthesizer->readWorkingSet(fileName,false);

    _synthesizer->setStringParameter("schedule_file", scheduleFile);
    _synthesizer->setStringParameter("tasks", taskString);

    _synthesizer->setOutputDir(outputDir);
    if (workingDir.isEmpty()) {
        _synthesizer->setStoreIntermediateImagesInTemp(false);
    } else {
        _synthesizer->setTemporaryDir(workingDir);
        _synthesizer->setStoreIntermediateImagesInTemp(true);
    }

    return success;
}

//------------------------------------------------------------------------------
// Client / server connection

bool NLMainWindow::startServer()
{
    if (_tcpServer)
        return false;

    _tcpServer = new QTcpServer(this);
    if (!_tcpServer->listen(QHostAddress::LocalHost, 55555)) {
        qCritical("Unable to start tcp server");
        return false;
    }

    connect(_tcpServer, SIGNAL(newConnection()), this, SLOT(handleSocketConnection()));
    qDebug("Started server on localhost 55555");
    return true;
}

bool NLMainWindow::stopServer()
{
    if (!_tcpServer)
        return false;

    delete _tcpServer;
    _tcpServer = NULL;
    qDebug("Server stopped.");

    return true;
}

void NLMainWindow::handleSocketConnection()
{
    QTcpSocket* socket = _tcpServer->nextPendingConnection();
    connect(socket, SIGNAL(disconnected()),
            socket, SLOT(deleteLater()));

    NLNetworkMessage request, reply;

    nlServerReceiveRequest(socket, &request);

    _currentNetworkRequest = request;
    // Call the handle functions with a timer so that the network code returns immediately.
    if (request["type"] == "start") {
        QTimer::singleShot(100, this, SLOT(handleStartRequest()));
    } else if (request["type"] == "styleRefresh") {
        QTimer::singleShot(100, this, SLOT(handleStyleRefreshRequest()));
    } else if (request["type"] == "finish") {
        QTimer::singleShot(100, this, SLOT(handleFinishRequest()));
    }

    reply["status"] = "started";
    nlServerSendReply(socket, reply);

    socket->disconnectFromHost();
}

void NLMainWindow::handleStartRequest()
{
    QString working_set = _currentNetworkRequest["working_set"];
    qDebug("Requested working set: %s\n", qPrintable(working_set));

    setRequestOptions();

    readWorkingSet(qPrintable(working_set), false);
    updateWindows();
    setupPostSynthesisHandling();
}

void NLMainWindow::handleStyleRefreshRequest()
{
    QString working_set = _currentNetworkRequest["working_set"];
    qDebug("Requested working set: %s\n", qPrintable(working_set));

    setRequestOptions();

    readWorkingSet(qPrintable(working_set), true);
    updateWindows();
    setupPostSynthesisHandling();
}

void NLMainWindow::handleFinishRequest()
{
    qDebug("Got request type: %s\n", qPrintable(_currentNetworkRequest["type"]));
}

void NLMainWindow::setRequestOptions()
{
    bool use_cached_preprocess = (_currentNetworkRequest["use_cached_keyframe_preprocess"] == "yes") ? true : false;
    NLParameters::instance().setBool("use_cached_preprocess", use_cached_preprocess);
}

void NLMainWindow::setupPostSynthesisHandling(){
    // Saving the image is done in the render loop itself now.
    _synthesizer->setOutputDir(QString(_synthesizer->getWorkingDir()+ "%1-%2_v%3/").arg(_synthesizer->firstFrame())
                               .arg(_synthesizer->lastFrame()).arg(_synthesizer->version()));
    _synthesizer->setStoreIntermediateImagesInTemp(true);
    _animationSaveDir = _synthesizer->getOutDir();
}
