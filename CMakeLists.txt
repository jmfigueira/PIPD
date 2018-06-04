cmake_minimum_required (VERSION 2.6)

PROJECT(PIPD)

# Requires OpenCV
FIND_PACKAGE( OpenCV REQUIRED )

# OpenCV config
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS})
ENDMACRO()


add_example(projetoDados)#cmake -G "Visual Studio 14 2015 Win64" ..