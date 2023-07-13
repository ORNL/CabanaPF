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
            std::cout << t << std::endl;
            Simulation<PFHub1aSimplePeriodic>(96, t).timestep(t);
            Simulation<PFHub1aBenchmark>(96, t).timestep(t);
        }
        for (int g=96; g<=6144; g*=2) {
            Simulation<PFHub1aSimplePeriodic>(g, 500).timestep(500);
            Simulation<PFHub1aBenchmark>(g, 500).timestep(500);
        }
    }
    MPI_Finalize();
    return 0;
}
