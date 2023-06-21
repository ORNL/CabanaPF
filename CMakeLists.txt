cmake_minimum_required(VERSION 3.20)
project(CabanaPF)
find_package(Cabana)
find_package(GTest)

add_subdirectory(examples)

add_subdirectory(unit_test)
