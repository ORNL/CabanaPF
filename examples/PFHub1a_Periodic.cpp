#include <PFHub.hpp>

using namespace CabanaPF;
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    {
        Kokkos::ScopeGuard scope_guard(argc, argv);
        if (argc == 1) {
            // runs for the paper
            for (int t = 500; t <= 1024000; t *= 2) {
                std::cout << "Running " << t << " timesteps" << std::endl;
                PFHub1aPeriodic simulation(96, t);
                simulation.timestep(t);
                simulation.output();
            }
            for (int g = 96; g <= 6144; g *= 2) {
                std::cout << "Running " << g << " grid points" << std::endl;
                PFHub1aPeriodic simulation(g, 500);
                simulation.timestep(500);
                simulation.output();
            }
        } else {
            try {
                int grid_points = std::stoi(argv[1]);
                int timesteps = std::stoi(argv[2]);
                bool should_write = (argc == 3); // TODO: flag?  Command line stuff is fairly temporary
                PFHub1aPeriodic simulation(grid_points, timesteps);
                simulation.timestep(timesteps);
                if (should_write)
                    simulation.output();
                std::cout << "done" << std::endl;
            } catch (std::logic_error const&) {
                std::cout << "Usage: ./Benchmark [grid_points] [timesteps]" << std::endl;
            }
        }
    }
    MPI_Finalize();
    return 0;
}
