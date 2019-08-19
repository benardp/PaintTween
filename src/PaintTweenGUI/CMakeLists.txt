include_directories(
   ${OPENGL_INCLUDE_DIR}
   ${OPENEXR_INCLUDE_DIR}
   ${PROJECT_SOURCE_DIR}/src
   ${Qt5Widgets_INCLUDE_DIRS}
   ${Qt5Xml_INCLUDE_DIRS} 
   ${Qt5OpenGL_INCLUDE_DIRS}
)

#-------------------------------------------------------------------------------
# source & headers

set(CPU_SOURCE_FILES
   main.cpp
   nlAnimationDialog.cpp
   nlConsoleInterface.cpp
   nlDataAccess.cpp
   nlDialsAndKnobs.cpp
   nlGLWidget.cpp
   nlImageContainer.cpp
   nlImageViewerWidget.cpp
   nlInputElements.cpp
   nlMainWindow.cpp
   nlParameters.cpp
   nlPlayRangeWidget.cpp
   nlShot.cpp
   nlStyles.cpp
   nlStyleTabWidget.cpp
   nlSynthesisWidget.cpp
   nlSynthesizer.cpp
   networkProtocol.cpp
   qcommandline.cpp
)

set(CPU_HEADER_FILES
   nlConsoleInterface.h
   nlDataAccess.h
   nlImageContainer.h
   nlInputElements.h
   nlShot.h
   nlStyles.h
   networkProtocol.h
)

set(CPU_MOC_HEADER_FILES
   nlAnimationDialog.h
   nlDialsAndKnobs.h
   nlGLWidget.h
   nlImageViewerWidget.h
   nlMainWindow.h
   nlParameters.h
   nlPlayRangeWidget.h
   nlStyleTabWidget.h
   nlSynthesisWidget.h
   nlSynthesizer.h
   qcommandline.h
)

SET(CPU_RESOURCES icons.qrc)

#-------------------------------------------------------------------------------
# platform dependent tweaks
if(APPLE OR UNIX)
    set(PLATFORM_COMPILE_FLAGS
        -fPIC
    )
endif()

QT5_ADD_RESOURCES(CPU_RESOURCES_RCC ${CPU_RESOURCES})

set(CMAKE_CXX_FLAGS ${PLATFORM_COMPILE_FLAGS})

add_executable( PaintTweenGUI 
   ${CPU_SOURCE_FILES}
   ${CPU_HEADER_FILES}
   ${CPU_RESOURCES_RCC}
)

target_link_libraries(PaintTweenGUI
   PaintTweenCUDA
   ${OPENGL_LIBRARY}
   ${OPENEXR_LIBRARIES}
   Qt5::Widgets
   Qt5::OpenGL
   Qt5::Xml
   Qt5::Network
   ${GLEW_LIBRARIES}
)

install( TARGETS DESTINATION ${CMAKE_LIBDIR_BASE} )
