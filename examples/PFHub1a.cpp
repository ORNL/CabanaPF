#include <Cajita.hpp>
#include <PFHub.hpp>

int main(int argc, char* argv[]) {
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );
        CabanaPF::PfHubProblem simulation(96);
        simulation.fill_initial();
        for (int i=0; i<10; i++) {
            std::cout << "Initial value at (" << i << "," << i << "): " << simulation.get_c(i, i) << std::endl;
        }
    }
    MPI_Finalize();
    std::cout << "done" << std::endl;
    return 0;
}