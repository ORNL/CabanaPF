cmake_minimum_required(VERSION 3.16)
project(CabanaPF)
find_package(Cabana)
find_package(GTest)

add_executable(CabanaPF main.cpp)
target_link_libraries(CabanaPF Cabana::cabanacore)

add_executable(Test test.cpp)
target_compile_definitions(Test PUBLIC BUILD_FOR_TESTS)
target_link_libraries(Test Cabana::cabanacore)
target_link_libraries(Test GTest::GTest)
