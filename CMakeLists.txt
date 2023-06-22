cmake_minimum_required(VERSION 3.20)
project(CabanaPF CXX)

include(GNUInstallDirs)

find_package(Cabana)
if(NOT Cabana_ENABLE_HEFFTE)
  message( FATAL_ERROR "Cabana must be compiled with heFFTe." )
endif()

find_package(GTest)

add_subdirectory(src)

add_subdirectory(examples)

add_subdirectory(unit_test)
