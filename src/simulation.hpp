#ifndef SIMULATION_H
#define SIMULATION_H

#include <Physics.hpp>
#include <Runner.hpp>

namespace CabanaPF {

//TODO: Template 2D vs 3D
template<typename Runner>
class Simulation {
    //static_assert(std::is_base_of<Runner, IRunner<2>>::value, "Runner must implement IRunner");
    //^ better than duck typing, since it gives clear error message

    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;
    using Mesh = Cajita::UniformMesh<double, 2>;
    using cdouble = std::complex<double>;
private:
    int timesteps_done;
    std::shared_ptr<Cajita::LocalGrid<Mesh>> local_grid;
    std::unique_ptr<Runner> runner;
public:
    const int grid_points;
    const int timesteps;

    Simulation(int grid_points, int timesteps)
        : grid_points{grid_points}, timesteps{timesteps}
    {
        //create global grid and mesh:
        auto global_mesh = Cajita::createUniformGlobalMesh(
            std::array<double, 2> {0, 0},
            std::array<double, 2> {Runner::SIZE, Runner::SIZE},  //TODO: Specify size in physics
            std::array<int, 2> {grid_points, grid_points}
        );
        Cajita::DimBlockPartitioner<2> partitioner;
        auto global_grid = Cajita::createGlobalGrid(MPI_COMM_WORLD, global_mesh, std::array<bool, 2>{true, true}, partitioner);

        //create local stuff:
        local_grid = Cajita::createLocalGrid(global_grid, 0);
        auto layout = createArrayLayout(local_grid, 2, Cajita::Cell()); //2: real & imag
        //create runner object and initialize variables
        runner = std::make_unique<Runner>(grid_points, timesteps, layout);
        Cajita::grid_parallel_for("initialize", exec_space(), *local_grid, Cajita::Ghost(), Cajita::Cell(), runner->initialize());
    }

    double get_c(int i, int j) {
        return runner->vars[0](i, j, 0);
    }

    void timestep(int count) {
        for(int i=0; i<count; i++) {
            const std::function<void(int, int)> pre_step = runner->pre_step();
            if (pre_step)   //don't run if nullptr
                Cajita::grid_parallel_for("pre_step", exec_space(), *local_grid, Cajita::Ghost(), Cajita::Cell(), pre_step);
            const std::function<void(int, int)> step = runner->step();
            if (step)
                Cajita::grid_parallel_for("step", exec_space(), *local_grid, Cajita::Ghost(), Cajita::Cell(), step);
            const std::function<void(int, int)> post_step = runner->post_step();
            if (post_step)
                Cajita::grid_parallel_for("post_step", exec_space(), *local_grid, Cajita::Ghost(), Cajita::Cell(), post_step);
        }
        timesteps_done += count;
    }
};

}

#endif
