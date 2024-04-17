#include <PFHub.hpp>

using namespace CabanaPF;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    {
        Kokkos::ScopeGuard scope_guard(argc, argv);
        try {
            if (argc != 5)
                throw std::invalid_argument("");
            const int grid_points = std::stoi(argv[2]);
            const double dt = std::stod(argv[3]);
            const double end_time = std::stod(argv[4]);

            // read in the type of simulation to run:
            std::unique_ptr<PFHub1aBase> simulation;
            std::string problem_name(argv[1]);
            if (problem_name == "benchmark")
                simulation = std::make_unique<PFHub1aBenchmark>(grid_points, dt);
            else if (problem_name == "periodic")
                simulation = std::make_unique<PFHub1aPeriodic>(grid_points, dt);
            else
                throw std::invalid_argument("");

            simulation->run_until_time(end_time);
            simulation->output();
        } catch (std::invalid_argument const&) {
            std::cout << "Usage: ./PFHub1a <benchmark|periodic> <grid points> <dt> <end time>" << std::endl;
        }
    }
    MPI_Finalize();
    return 0;
}
