#include <Cajita.hpp>
#include <simulation.hpp>
#include <PFVariables.hpp>
#include <PFHub.hpp>

using namespace CabanaPF;
int main(int argc, char* argv[]) {
    MPI_Init( &argc, &argv );
    {
        //TODO: Make these changeable (file I/O)
        int grid_points = 96;
        int timesteps = 500;
        Kokkos::ScopeGuard scope_guard( argc, argv );
        //runs for the paper
        for (int t=500; t<=1024000; t*=2) {
            Simulation<PFHub1aFixedPeriodic>(96, t).timestep(t);
            std::cout << t << std::endl;
        }
        for (int g=96; g<=6144; g*=2) {
            Simulation<PFHub1aFixedPeriodic>(g, 500).timestep(500);
            std::cout << g << std::endl;
        }
    }
    MPI_Finalize();
    return 0;
}
