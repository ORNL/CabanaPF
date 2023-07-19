#ifndef RUNNER_H
#define RUNNER_H

#include <Cajita.hpp>

namespace CabanaPF {

//Inherit from this to implement the problem's specific actions
template <std::size_t NumSpaceDim>
class CabanaPFRunner {
protected:
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;
    using Mesh = Cajita::UniformMesh<double, 2>;

    std::shared_ptr<Cajita::LocalGrid<Mesh>> local_grid;
    std::shared_ptr<Cajita::ArrayLayout<Cajita::Node, Mesh>> layout;
    int timesteps_done;
    bool have_initialized;
public:
    const int grid_points;
    const int timesteps;

    CabanaPFRunner(int grid_points, int timesteps, double size)
        : grid_points{grid_points}, timesteps{timesteps}, timesteps_done{0}, have_initialized{false}
    {
        auto global_mesh = Cajita::createUniformGlobalMesh(
            std::array<double, 2> {0, 0},
            std::array<double, 2> {size, size},
            std::array<int, 2> {grid_points, grid_points}
        );
        Cajita::DimBlockPartitioner<2> partitioner;
        auto global_grid = Cajita::createGlobalGrid(MPI_COMM_WORLD, global_mesh, std::array<bool, 2>{true, true}, partitioner);

        //create local stuff:
        local_grid = Cajita::createLocalGrid(global_grid, 0);
        layout = createArrayLayout(local_grid, 2, Cajita::Node()); //2: real & imag
    }

    template<class FunctorType>
    void parallel_for(const std::string& label, FunctorType lambda) {
        Cajita::grid_parallel_for(label, exec_space(), *local_grid, Cajita::Own(), Cajita::Node(), lambda);
    }

    template<class FunctorType>
    double parallel_reduce(const std::string& label, FunctorType lambda) {
        double result = 0;
        Cajita::grid_parallel_reduce(label, exec_space(), *local_grid, Cajita::Own(), Cajita::Node(), lambda, result);
        return result;
    }

    void timestep(int count) {
        if (!have_initialized) {
            initialize();
            have_initialized = true;
        }
        for(int i=0; i<count; i++) {
            pre_step();
            step();
            post_step();
            timesteps_done++;
        }
        if (timesteps_done==timesteps)
            finalize();
    }

    //Generally, you inherit from this class and implement one or more of these:
    virtual void initialize() {} //Called once, before taking the first timestep
    virtual void pre_step() {}   //Called each timestep
    virtual void step() {}       //Called each timestep, after pre_step
    virtual void post_step() {}  //Called each timestep, after step
    virtual void finalize() {}   //Called once, when the requested number of timesteps have been done
};

}

#endif
