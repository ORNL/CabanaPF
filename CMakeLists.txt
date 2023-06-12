cmake_minimum_required(VERSION 3.16)
project(CabanaPF)
find_package(Cabana)

add_executable(CabanaPF main.cpp)
target_link_libraries(CabanaPF Cabana::cabanacore)
