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

#include <iostream>
#include <QtNetwork/QTcpSocket>
#include <QtNetwork/QHostAddress>

#include "networkProtocol.h"

const int timeout = 1000;

bool readNetworkMessage(QTcpSocket* socket, NLNetworkMessage* message)
{
    while (socket->bytesAvailable() < (int)sizeof(quint16)) {
        if (!socket->waitForReadyRead(timeout)) {
            std::cerr << "socket error: " << qPrintable(socket->errorString()) << "\n";
            return false;
        }
    }

    quint16 blockSize;
    QDataStream in(socket);
    in.setVersion(QDataStream::Qt_4_0);
    in >> blockSize;

    while (socket->bytesAvailable() < blockSize) {
        if (!socket->waitForReadyRead(timeout)) {
            std::cerr << "socket error (2): " << qPrintable(socket->errorString()) << "\n";
            return false;
        }
    }

    in >> *message;
    return true;
}

bool writeNetworkMessage(QTcpSocket* socket, const NLNetworkMessage& message)
{
    QByteArray block;
    QDataStream out(&block, QIODevice::WriteOnly);
    out.setVersion(QDataStream::Qt_4_0);
    out << (quint16)0;
    out << message;
    out.device()->seek(0);
    out << (quint16)(block.size() - sizeof(quint16)); 

    int bytes_written = socket->write(block);
    if (bytes_written != block.size()) {
        std::cerr << "socket write error, tried to write " << block.size() << " but only wrote " << bytes_written << "\n";
        return false;
    } 
    return true;
}

bool nlClientSendAndReceive(const NLNetworkMessage& request, NLNetworkMessage* reply)
{
    QString ip_address = QHostAddress(QHostAddress::LocalHost).toString();
    QTcpSocket socket;
    socket.connectToHost(ip_address, 55555);
    if (!socket.waitForConnected(timeout)) {
        std::cerr << "socket connection error: " << qPrintable(socket.errorString()) << "\n";
        return false;
    }

    bool write_success = writeNetworkMessage(&socket, request); 

    bool read_success = readNetworkMessage(&socket, reply);

    socket.waitForDisconnected(timeout);
    
    return write_success && read_success;
}

bool nlServerReceiveRequest(QTcpSocket* socket, NLNetworkMessage* request)
{
    return readNetworkMessage(socket, request);
}

bool nlServerSendReply(QTcpSocket* socket, const NLNetworkMessage& reply)
{
    return writeNetworkMessage(socket, reply);
}
