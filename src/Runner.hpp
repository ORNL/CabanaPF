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
    using Mesh = Cajita::UniformMesh<double, NumSpaceDim>;

    std::shared_ptr<Cajita::LocalGrid<Mesh>> local_grid;
    std::shared_ptr<Cajita::ArrayLayout<Cajita::Node, Mesh>> layout;
    int timesteps_done;
    bool have_initialized;
public:
    const int grid_points;
    const int timesteps;

    CabanaPFRunner(int grid_points, int timesteps, double size)
        : timesteps_done{0}, have_initialized{false}, grid_points{grid_points}, timesteps{timesteps}
    {
        std::array<double, NumSpaceDim> low_corner;
        low_corner.fill(0.0);
        std::array<double, NumSpaceDim> high_corner;
        high_corner.fill(size);
        std::array<int, NumSpaceDim> num_cell;
        num_cell.fill(grid_points);
        auto global_mesh = Cajita::createUniformGlobalMesh(
            low_corner, high_corner, num_cell
        );
        Cajita::DimBlockPartitioner<NumSpaceDim> partitioner;
        std::array<bool, NumSpaceDim> periodic;
        periodic.fill(true);
        auto global_grid = Cajita::createGlobalGrid(MPI_COMM_WORLD, global_mesh, periodic, partitioner);

        //create local stuff:
        local_grid = Cajita::createLocalGrid(global_grid, 0);
        layout = createArrayLayout(local_grid, 2, Cajita::Node()); //2: real & imag
    }

    template<class FunctorType>
    void node_parallel_for(const std::string& label, FunctorType lambda) {
        Cajita::grid_parallel_for(label, exec_space(), *local_grid, Cajita::Own(), Cajita::Node(), lambda);
    }

    template<class FunctorType>
    double node_parallel_reduce(const std::string& label, FunctorType lambda) {
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
