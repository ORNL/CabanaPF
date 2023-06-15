include_directories(../src)
add_executable(Test test.cpp)
target_link_libraries(Test Cabana::cabanacore)
target_link_libraries(Test GTest::GTest)
