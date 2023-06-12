#include <iostream>
#include <Cabana_Core.hpp>
#include <Cajita.hpp>

int main( int argc, char* argv[] ) {
    Kokkos::ScopeGuard scope_guard( argc, argv );

    std::cout << "hello world" << std::endl;
    std::cout << "Cabana version: " << Cabana::version() << std::endl;
}
