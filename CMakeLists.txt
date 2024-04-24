cmake_minimum_required(VERSION 3.16)
project(CabanaPF CXX)

include(GNUInstallDirs)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(DEFINED CABANAPF_RESULTS_PATH)
  add_compile_definitions(CABANAPF_RESULTS_PATH="${CABANAPF_RESULTS_PATH}")
endif()

find_package(Cabana)
if(NOT Cabana_ENABLE_HEFFTE)
  message( FATAL_ERROR "Cabana must be compiled with heFFTe." )
endif()

find_package(GTest)
# Workaround for FindGTest module in CMake older than 3.20
if(TARGET GTest::gtest)
  set(gtest_target GTest::gtest)
elseif(TARGET GTest::GTest)
  set(gtest_target GTest::GTest)
else()
  message(FATAL_ERROR "bug in GTest find module workaround")
endif()

add_subdirectory(src)

add_subdirectory(examples)

enable_testing()
add_subdirectory(unit_test)

## Clang Format
find_package( CLANG_FORMAT 14 )
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.[c,h]pp unit_test/*.[c,h]pp examples/*.[c,h]pp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
