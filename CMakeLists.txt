cmake_minimum_required(VERSION 3.20)
project(CabanaPF CXX)

include(GNUInstallDirs)

if(DEFINED RESULTS_PATH)
  add_compile_definitions(RESULTS_PATH="${RESULTS_PATH}")
endif()

find_package(Cabana)
if(NOT Cabana_ENABLE_HEFFTE)
  message( FATAL_ERROR "Cabana must be compiled with heFFTe." )
endif()

find_package(GTest)

add_subdirectory(src)

add_subdirectory(examples)

enable_testing()
add_subdirectory(unit_test)
