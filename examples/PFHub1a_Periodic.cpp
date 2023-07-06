#include <Cajita.hpp>
#include <simulation.hpp>
#include <PFVariables.hpp>
#include <PFHub.hpp>

using namespace CabanaPF;
int main(int argc, char* argv[]) {
    //TODO: Read from command line
    int grid_points = 96;
    int timesteps = 500;
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );
        Simulation<PFHub1aPeriodic> simul(grid_points, timesteps);
        simul.timestep(timesteps);
        //TODO: File write
    }
    MPI_Finalize();
    return 0;
}
