file(GLOB HEADERS GLOB *.hpp)

add_library(CabanaPF INTERFACE)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

target_include_directories(CabanaPF INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

target_link_libraries(CabanaPF INTERFACE Cabana::Grid)

install(TARGETS CabanaPF DESTINATION ${CMAKE_INSTALL_LIBDIR})
