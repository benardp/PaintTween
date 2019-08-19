project(texSynth)
enable_language(CUDA)

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
#set(CUDA_BUILD_TYPE  Device)
#set(CUDA_NVCC_FLAGS "")

#find_package(CUDA REQUIRED)

include_directories(
	${CUDA_INSTALL_PREFIX}/include
	${GLEW_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}
    ${QT_INCLUDES}
    ${CUDA_INSTALL_PREFIX}/include
)
#set(CUDA_NVCC_FLAGS -use_fast_math -arch sm_20)# -arch sm_20) #-m64

add_subdirectory(src)
