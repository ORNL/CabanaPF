#ifndef RUNNER_H
#define RUNNER_H

#include <Cabana_Grid.hpp>

namespace CabanaPF {

// Inherit from this to implement the problem's specific actions
template <std::size_t NumSpaceDim>
class CabanaPFRunner {
  protected:
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;
    using Mesh = Cabana::Grid::UniformMesh<double, NumSpaceDim>;

    std::shared_ptr<Cabana::Grid::LocalGrid<Mesh>> local_grid;
    std::shared_ptr<Cabana::Grid::ArrayLayout<Cabana::Grid::Node, Mesh>> layout;
    int timesteps_done;
    bool have_initialized;

  public:
    const int grid_points;
    const int timesteps;

    CabanaPFRunner(int grid_points, int timesteps, double size)
        : timesteps_done{0}, have_initialized{false}, grid_points{grid_points}, timesteps{timesteps} {
        std::array<double, NumSpaceDim> low_corner;
        low_corner.fill(0.0);
        std::array<double, NumSpaceDim> high_corner;
        high_corner.fill(size);
        std::array<int, NumSpaceDim> num_cell;
        num_cell.fill(grid_points);
        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(low_corner, high_corner, num_cell);
        Cabana::Grid::DimBlockPartitioner<NumSpaceDim> partitioner;
        std::array<bool, NumSpaceDim> periodic;
        periodic.fill(true);
        auto global_grid = Cabana::Grid::createGlobalGrid(MPI_COMM_WORLD, global_mesh, periodic, partitioner);

        // create local stuff:
        local_grid = Cabana::Grid::createLocalGrid(global_grid, 0);
        layout = createArrayLayout(local_grid, 2, Cabana::Grid::Node()); // 2: real & imag
    }

    template <class FunctorType>
    void node_parallel_for(const std::string& label, FunctorType lambda) {
        Cabana::Grid::grid_parallel_for(label, exec_space(), *local_grid, Cabana::Grid::Own(), Cabana::Grid::Node(),
                                        lambda);
    }

    template <class FunctorType>
    double node_parallel_reduce(const std::string& label, FunctorType lambda) {
        double result = 0;
        Cabana::Grid::grid_parallel_reduce(label, exec_space(), *local_grid, Cabana::Grid::Own(), Cabana::Grid::Node(),
                                           lambda, result);
        return result;
    }

    void timestep(int count) {
        if (!have_initialized) {
            initialize();
            have_initialized = true;
        }
        for (int i = 0; i < count; i++) {
            pre_step();
            step();
            post_step();
            timesteps_done++;
        }
        if (timesteps_done == timesteps)
            finalize();
    }

    // Generally, you inherit from this class and implement one or more of these:
    virtual void initialize() {} // Called once, before taking the first timestep
    virtual void pre_step() {}   // Called each timestep
    virtual void step() {}       // Called each timestep, after pre_step
    virtual void post_step() {}  // Called each timestep, after step
    virtual void finalize() {}   // Called once, when the requested number of timesteps have been done
};

} // namespace CabanaPF

#endif
