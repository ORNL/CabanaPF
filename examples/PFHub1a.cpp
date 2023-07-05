#include <Cajita.hpp>
#include <PFHub.hpp>

//int TIMESTEPS[8] = {500, 1000, 2000, 4000, 8000, 16000, 32000, 64000};
int TIMESTEPS[1] = {2000};
int GRIDS[1] = {16*96};

int main(int argc, char* argv[]) {
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );
        for(int steps : TIMESTEPS) {
            std::cout << "Running " << steps << std::endl;
            CabanaPF::PfHubProblem simulation(96);
            simulation.fill_initial();
            simulation.timestep(250./steps, steps);
            std::cout << "Finished " << steps << std::endl;
        }
        /*for(int grid : GRIDS) {
            std::cout << "Running " << grid << std::endl;
            CabanaPF::PfHubProblem simulation(grid);
            simulation.fill_initial();
            simulation.timestep(.5, 500);
            std::cout << "Finished " << grid << std::endl;
        }*/
    }
    MPI_Finalize();
    std::cout << "done" << std::endl;
    return 0;
}
