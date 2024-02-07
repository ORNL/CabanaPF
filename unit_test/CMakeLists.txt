add_executable(CabanaPF_Test_PFHub1a test.cpp)
target_link_libraries(CabanaPF_Test_PFHub1a LINK_PUBLIC CabanaPF ${gtest_target})
add_test(NAME PFHub1a COMMAND ${NONMPI_PRECOMMAND} CabanaPF_Test_PFHub1a)
