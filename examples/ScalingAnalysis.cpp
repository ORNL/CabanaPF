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
        double dt = 0, end_time = 0;
        try {
            if (argc < 4)
                // Need at least: Executable name, timesteps, end time, one grid points
                throw std::invalid_argument("");
            dt = std::stoi(argv[2]);
            end_time = std::stoi(argv[3]);
            for (int i = 3; i < argc; i++)
                runs.push_back(std::stoi(argv[i]));
        } catch (std::invalid_argument const&) {
            std::cout << "Usage: ./ScalingAnalysis <dt> <end time> <grid points> [grid points ...]" << std::endl;
            return 1;
        }

        // run these and measure how long they take:
        Cabana::Benchmark::Timer timer("grid scaling", runs.size());
        for (std::size_t i = 0; i < runs.size(); i++) {
            std::cout << "Running " << runs[i] << " grid points" << std::endl;
            for (int reps = 0; reps < 5; reps++) {
                timer.start(i);
                PFHub1aCHiMaD2023 simul(runs[i], dt);
                simul.run_until_time(end_time);
                timer.stop(i);
            }
        }

        // print results:
        Cabana::Benchmark::outputResults(std::cout, "grid points", runs, timer);
    }
    MPI_Finalize();
    return 0;
}
