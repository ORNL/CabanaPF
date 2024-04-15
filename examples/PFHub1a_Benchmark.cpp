#include <PFHub.hpp>

using namespace CabanaPF;
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    {
        Kokkos::ScopeGuard scope_guard(argc, argv);
        if (argc == 1) {
            // runs for the paper
            for (int t = 2; t <= 2048; t *= 2) {
                std::cout << "Running " << t << " timesteps" << std::endl;
                PFHub1aBenchmark simulation(96, t);
                simulation.run(250 * t);
                simulation.output();
            }
            for (int g = 96; g <= 6144; g *= 2) {
                std::cout << "Running " << g << " grid points" << std::endl;
                PFHub1aBenchmark simulation(g, 2);
                simulation.run(500);
                simulation.output();
            }
        } else {
            try {
                int grid_points = std::stoi(argv[1]);
                int timesteps_per_t = std::stoi(argv[2]);
                bool should_write = (argc == 3); // TODO: flag?  Command line stuff is fairly temporary
                PFHub1aBenchmark simulation(grid_points, timesteps_per_t);
                simulation.run(250 * timesteps_per_t);
                if (should_write)
                    simulation.output();
                std::cout << "done" << std::endl;
            } catch (std::logic_error const&) {
                std::cout << "Usage: ./Benchmark grid_points timesteps [no file write]" << std::endl;
            }
        }
    }
    MPI_Finalize();
    return 0;
}
