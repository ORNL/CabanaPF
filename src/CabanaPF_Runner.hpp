#ifndef RUNNER_H
#define RUNNER_H

#include <Cabana_Grid.hpp>
#include <queue>

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

    using MinHeap = std::priority_queue<int, std::vector<int>, std::greater<int>>;
    MinHeap output_steps_major;
    MinHeap output_steps_minor;

  public:
    const int grid_points;
    const double dt;

    CabanaPFRunner(int grid_points, double size, double dt)
        : timesteps_done{0}, have_initialized{false}, grid_points{grid_points}, dt{dt} {
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

    // Returns the number of timesteps closest to the given time, based on dt
    int closest_timesteps(const double time) {
        return round(time / dt);
    }

    // Do a certain number of timesteps:
    void run_for_steps(const int timesteps) {
        if (!have_initialized) {
            initialize();
            have_initialized = true;
        }
        for (int i = 0; i < timesteps; i++) {
            step();
            timesteps_done++;
            // check if we output this timestep:
            while (!output_steps_major.empty() && output_steps_major.top() == timesteps_done) {
                output_steps_major.pop();
                major_output();
            }
            while (!output_steps_minor.empty() && output_steps_minor.top() == timesteps_done) {
                output_steps_minor.pop();
                minor_output();
            }
        }
    }

    // runs the closest number of timesteps to time/dt
    void run_for_time(const double time) {
        run_for_steps(closest_timesteps(time));
    }

    // runs as many timesteps as are needed to have done a certain number
    void run_until_steps(const int timesteps) {
        run_for_steps(timesteps - timesteps_done);
    }

    // runs until timesteps_done*dt is as close as possible to specified time
    void run_until_time(const double time) {
        run_for_steps(closest_timesteps(time) - timesteps_done);
    }

    int get_timesteps_done() const {
        return timesteps_done;
    }

    double get_time_done() const {
        return dt * timesteps_done;
    }

    void add_output(double time, bool is_major) {
        int step = closest_timesteps(time);
        if (step < timesteps_done)
            throw std::logic_error("Attempted to add output prior to current simulation time");
        else if (step == timesteps_done) {
            // user's output at t=0 likely depends on PF variables, so we need to initialize if so:
            if (timesteps_done == 0 && !have_initialized) {
                initialize();
                have_initialized = true;
            }
            if (is_major)
                major_output();
            else
                minor_output();
        } else {
            if (is_major)
                output_steps_major.push(step);
            else
                output_steps_minor.push(step);
        }
    }

    // Children inherit from this class and implement these:
    virtual void initialize() {}   // Called once, before taking the first timestep
    virtual void step() {}         // Called to take a timestep
    virtual void major_output() {} // Called a number of times based on user's command line arguments
    virtual void minor_output() {
    } // As above, but is a lighter operation (like printing free energy instead of writing a grid)
};

} // namespace CabanaPF

#endif
