include_directories(
   ${OPENGL_INCLUDE_DIR}
   ${OPENEXR_INCLUDE_DIR}
   ${Qt5Widgets_INCLUDE_DIRS}
   ${Qt5Xml_INCLUDE_DIRS} 
   ${Qt5OpenGL_INCLUDE_DIRS}
)

#-------------------------------------------------------------------------------
# source & headers

set(CPU_SOURCE_FILES
   advectedBuffers.cpp
   batchProcessor.cpp
   cudaHostUtil.cpp
   cudaImagePyramid.cpp
   cudaTexture.cpp
   dataAccess.cpp
   downsample.cpp
   imageIO.cpp
   imagePyramid.cpp
   stats.cpp
   style.cpp
   synthesisFrame.cpp
   synthesisProcessor.cpp
   workingBuffers.cpp
   workSchedule.cpp
)

set(CPU_HEADER_FILES
   advectedBuffers.h
   batchProcessor.h
   convolution.h
   cudaHostUtil.h
   cudaImageBuffer.h
   cudaImagePyramid.h
   cudaTexture.h
   cutil_math.h
   dataAccess.h
   downsample.h
   imageIO.h
   imagePyramid.h
   progressCallback.h
   stats.h
   style.h
   synthesisFrame.h
   synthesisProcessor.h
   texSynth_interface.h
   texSynth_kernel.h
   texSynth_util.h
   types.h
   workingBuffers.h
   workSchedule.h
)

set(CUDA_HEADER_FILE
    texSynth_util.h
    texSynth_kernel.h
    texSynth_interface.h
    cudaImageBuffer.h
    cudaImagePyramid.h
    convolution.h
    workingBuffers.h
    types.h
)

set(CUDA_KERNEL_FILES
   texSynth_kernel.cu
   convolution.cu
   workingBuffers.cu
)

#-------------------------------------------------------------------------------
# platform dependent tweaks
if(APPLE OR UNIX)
    set(PLATFORM_COMPILE_FLAGS
        -fPIC
    )
endif()

set(CMAKE_CXX_FLAGS ${PLATFORM_COMPILE_FLAGS})

#QT5_WRAP_CPP(CPU_HEADERS_MOC stats.h)

add_library(PaintTweenCUDA STATIC 
   ${CPU_SOURCE_FILES}
   ${CPU_HEADER_FILES}
   ${CUDA_KERNEL_FILES}
   ${CUDA_HEADER_FILES}
)

target_link_libraries(PaintTweenCUDA
   ${OPENGL_LIBRARY}
   ${OPENEXR_LIBRARIES}
   Qt5::Widgets
   Qt5::OpenGL
   ${GLEW_LIBRARIES}
)

install( TARGETS DESTINATION ${CMAKE_LIBDIR_BASE} )
