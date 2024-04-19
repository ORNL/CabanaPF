add_executable(PFHub1a PFHub1a.cpp)
target_link_libraries(PFHub1a LINK_PUBLIC CabanaPF)
add_executable(PerformanceRuns GridTimer.cpp)
target_link_libraries(PerformanceRuns LINK_PUBLIC CabanaPF)
