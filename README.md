# PaintTween

This is the source code package for the paper:

Stylizing Animation by Example
by Pierre Benard, Forrester Cole, Michael Kass, Igor Mordatch, James Hegarty, 
Martin Sebastian Senn, Kurt Fleischer, Davide Pesare, Katherine Breeden
Pixar Technical Memo #13-03 (presented at SIGGRAPH 2013)

The original paper, source, and data are available here:

http://graphics.pixar.com/library/ByExampleStylization/

Copyright (c) 2013 by Disney-Pixar

Permission is hereby granted to use this software solely for 
non-commercial applications and purposes including academic or 
industrial research, evaluation and not-for-profit media
production.  All other rights are retained by Pixar.  For use 
for or in connection with commercial applications and
purposes, including without limitation in or in connection 
with software products offered for sale or for-profit media
production, please contact Pixar at tech-licensing@pixar.com.


## INTRODUCTION

This is the source code used to make the results in the "Stylizing Animation
by Example" paper. It has been slightly modified to make it work outside of the
Pixar environment. If you have also downloaded the accompanying working set
images, you should be able to recreate the results of the paper using this code.

This is *research code*. It has crash bugs, incomplete features, and minimal 
documentation. It is not intended for use in any kind of production environment.
It is intended to allow other researchers to build on the work in the paper.
That said, you can also make some pretty nice animations with it. 


## REQUIREMENTS

You need at least the following to run this code:

- Linux, Mac OS X or Windows (tested on Windows 10, with Visual Studio 2017)
- A CUDA-compatible GPU, capable of shader model 20. Fermi-architecture cards
   such as the Quadro 5000 or Geforce GTX580 work well.
- CUDA 5.5 or later (and the corresponding version of gcc on Linux -- tested with Cuda 10.1)
- Qt 5.x (tested with Qt 5.13)
- OpenEXR 1.6 or later (tested with OpenEXR 2.2)


## BUILDING

The code has two build systems: cmake and qmake. CMake should automatically
find the required libraries, but qmake needs to be told. 

### CMake

To build using cmake (at root directory):

```
cmake .
make
```

On Mac OS X, you may have to manually request CMake adopt gcc instead of clang,
using the following environment variables:

```
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
```

You may also need to manually add the location of libcudart.dylib to your
`DYLD_LIBRARY_PATH` to PaintTweenGUI.

### QMake

To build using qmake (at root directory), edit paths.pri to match the installed 
locations of nvcc, openexr, and Qt. Then:

```
qmake -r
make
```

This should produce a bin/PaintTweenGUI.


## RUNNING

To run:

```
cd bin
./PaintTweenGUI
```

Once the interface loads, try:

1. File->Read Working Set...
2. Choose one of the example working set files, such as "paint.xml"
3. To synthesize a single frame, hit the "refresh" button on the far left 
   of the toolbar.
4. To synthesize the animation, select "Animation" in the toolbar, then
   hit the "refresh" button. 


July 17 2013 - Modified August 19 2019
