#include <PFHub.hpp>

#include <Cabana_BenchmarkUtils.hpp>

#include <cmath>

using namespace CabanaPF;
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    {
        Kokkos::ScopeGuard scope_guard(argc, argv);

        // read in grid points to run off the command line:
        std::vector<int> runs;
        try {
            for (int i = 1; i < argc; i++)
                runs.push_back(std::stoi(argv[i]));
        } catch (std::logic_error const&) {
            std::cout << "Usage: ./GridTimer [grid_points] [grid_points] [...]" << std::endl;
            ;
            return 1;
        }

        // run these and measure how long they take:
        Cabana::Benchmark::Timer timer("grid scaling", runs.size());
        for (std::size_t i = 0; i < runs.size(); i++) {
            std::cout << "Running " << runs[i] << " grid points" << std::endl;
            for (int reps = 0; reps < 5; reps++) {
                timer.start(i);
                PFHub1aPeriodic simul(runs[i], 500);
                simul.timestep(500);
                timer.stop(i);
            }
        }

        // print results:
        Cabana::Benchmark::outputResults(std::cout, "grid points", runs, timer);
    }
    MPI_Finalize();
    return 0;
}
