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

#ifndef _STATS_H_
#define _STATS_H_

#include <QtCore/QString>
#include <QtCore/QList>
#include <QtCore/QTime>
#include <vector>
#include <QtCore/QAbstractItemModel>
#include <QtGui/QDockWidget>

class QMenu;
class QMainWindow;

class Stats : public QAbstractItemModel
{
    Q_OBJECT

  public:
    // Removes everything.
    void clear();
    // Resets all timers and counters to zero, but leaves them in the lists.
    void reset(); 

    // Emits an update signal for attached views.
    void updateView();

    void startTimer( const QString& name );
    void stopTimer( const QString& name );

    void setCounter( const QString& name, float value );
    void addToCounter( const QString& name, float value );

    void beginConstantGroup( const QString& name );
    void setConstant( const QString& name, float value );
    void setConstant( const QString& name, const QString& value );
    void endConstantGroup();

    int numTimers() const { return _records[TIMER].size(); }
    const QString& timerName( int which ) const 
        { return _records[TIMER][which].name; }
    float timerValue( int which ) const 
        { return _records[TIMER][which].value; }
    
    int numCounters() const { return _records[COUNTER].size(); }
    const QString& counterName( int which ) const 
        { return _records[COUNTER][which].name; }
    float counterValue( int which ) const 
        { return _records[COUNTER][which].value; }

    QString timerStatistics( const QString& name );
    QString allTimerStatistics();

    // implementation of QAbstractItemModel
    QVariant data( const QModelIndex& index, int role ) const;
    Qt::ItemFlags flags( const QModelIndex& index ) const;
    QVariant headerData( int section, Qt::Orientation orientation, 
            int role = Qt::DisplayRole) const;
    QModelIndex index( int row, int column, 
            const QModelIndex& parent = QModelIndex() ) const;
    QModelIndex parent(const QModelIndex &index) const;
    int rowCount( const QModelIndex& parent = QModelIndex() ) const;
    int columnCount( const QModelIndex& parent = QModelIndex() ) const;

    static Stats& instance() { return _global_instance; }

  protected:
    enum Category { TIMER, COUNTER, CONSTANT, NUM_CATEGORIES };
    class Record
    {
      public:
        Record() : name(), category(NUM_CATEGORIES), stamp(), 
                   value(0), str_value(), touches_since_last_reset(0),
                   parent(0), children() {}
      public:
        QString     name;
        Category    category;
        QTime       stamp;
        float       value;
        QString     str_value;
        int         touches_since_last_reset;

        Record*         parent;
        QList<Record*>  children;
    };
    

  protected:
    // Singleton class.
    Stats() { init(); }
    bool init();

    void clearCategory( Category which );
    void setChildValuesToZero( Record* record );

    int findTimer( const QString& name, const Record* parent );
    int findTimer( const Record* pointer );
    int findCounter( const QString& name );

    void removeTimer( int which );
    void removeCounter( int which );

  protected:
    QList<Record>       _records[NUM_CATEGORIES];
    Record              _headers[NUM_CATEGORIES];

    QList<Record*>      _timer_stack;
    QList<Record*>      _constant_stack;

    Record              _dummy_root;

    bool                _layout_changed;
    bool                _data_changed;

    static Stats        _global_instance;
};

class StatsWidget : public QDockWidget
{
    Q_OBJECT

  public:
    StatsWidget(QMainWindow* parent, QMenu* menu);

};



//
// Helper classes, functions, and defines
//

// When this object is created, it starts a timer.
// When it passes out of scope, it stops the timer.
class ScopeTimer
{
public:
    ScopeTimer( const QString& name ) { _name = name; Stats::instance().startTimer(name); }
    ~ScopeTimer() { Stats::instance().stopTimer(_name); }
protected:
    QString _name;
};

#ifndef DEMOUTILS_NO_TIMERS
#define __START_TIMER(X) Stats::instance().startTimer(X);
#define __STOP_TIMER(X) Stats::instance().stopTimer(X);
#define __SET_COUNTER(X,Y) Stats::instance().setCounter((X),(Y));
#define __ADD_TO_COUNTER(X,Y) Stats::instance().addToCounter((X),(Y));
#define __TIME_CODE_BLOCK(X) ScopeTimer __scope_timer(X);
#else
#define __START_TIMER(X) ;
#define __STOP_TIMER(X) ;
#define __SET_COUNTER(X,Y) ;
#define __ADD_TO_COUNTER(X,Y) ;
#define __TIME_CODE_BLOCK(X) ; 
#endif

#endif // _STATS_H_