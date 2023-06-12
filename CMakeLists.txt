cmake_minimum_required(VERSION 3.16)
project(CabanaPF)
if(NOT TARGET Cabana::cabanacore)
  find_package(Cabana)
endif()

add_executable(CabanaPF main.cpp)
target_link_libraries(CabanaPF Cabana::cabanacore)
