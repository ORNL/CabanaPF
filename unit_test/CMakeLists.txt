include_directories(../src)
add_executable(Test test.cpp)
target_link_libraries(Test Cabana::Cajita)
target_link_libraries(Test GTest::GTest)
