project(texSynth)

cmake_minimum_required(VERSION 2.8.12)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Specify the list of directories to search for cmake modules.
set(CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Xml REQUIRED)
find_package(Qt5Network REQUIRED)
find_package(Qt5OpenGL REQUIRED)

find_package(OpenGL REQUIRED)
find_package(OpenEXR REQUIRED)
find_package(GLEW REQUIRED)

IF (WIN32)
	ADD_DEFINITIONS(-DOPENEXR_DLL)
ENDIF ()

# CUDA 
set(CUDA_BUILD_CUBIN OFF)
set(CUDA_BUILD_TYPE  Device)
set(CUDA_NVCC_FLAGS "")

find_package(CUDA REQUIRED)

include_directories(
	${CUDA_INSTALL_PREFIX}/include
	${GLEW_INCLUDE_DIRS}
	)
cuda_include_directories(
    ${CMAKE_BINARY_DIR}
    ${QT_INCLUDES}
    ${CUDA_INSTALL_PREFIX}/include
)
set(CUDA_NVCC_FLAGS -use_fast_math -arch sm_20) #-m64

# Macro for adding a cuda executable if cuda is found and a regular
# executable otherwise.
macro(_add_possibly_cuda_executable target)
    if(CUDA_FOUND)
        cuda_add_executable(${target} ${ARGN})
    else()
        add_executable(${target} ${ARGN})
    endif()
endmacro()


# Macro for adding a cuda library if cuda is found and a regular
# library otherwise.
macro(_add_possibly_cuda_library target)
    if(CUDA_FOUND)
        cuda_add_library(${target} ${ARGN})
    else()
        add_library(${target} ${ARGN})
    endif()
endmacro()

add_subdirectory(src)
