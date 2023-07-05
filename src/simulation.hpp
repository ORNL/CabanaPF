#ifndef SIMULATION_H
#define SIMULATION_H

#include <Physics.hpp>

namespace CabanaPF {

template<std::size_t NumSpaceDim, std::size_t NumVariables>
class Simulation {
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;
    using Mesh = Cajita::UniformMesh<double, NumSpaceDim>;
    using cdouble = std::complex<double>;
private:
    int timesteps_done;
    //CabanaPF::Physics<NumSpaceDim, NumVariables> physics; //TODO: Figure out abstract type
    CabanaPF::PFHubOne physics;
    std::shared_ptr<Cajita::LocalGrid<Mesh>> local_grid;
    std::shared_ptr<Cajita::Experimental::HeffteFastFourierTransform<
        Cajita::Cell, Mesh, double, device_type, Cajita::Experimental::Impl::FFTBackendDefault>> fft;
    PFVariables<NumSpaceDim, NumVariables> phis;
    PFVariables<NumSpaceDim, NumVariables> df_dphis;

    Kokkos::View<cdouble **, device_type> laplacian;
    void setup_laplacian() {
        laplacian = Kokkos::View<cdouble **> ("laplacian", grid_points, grid_points);
        Cajita::grid_parallel_for(
            "laplacian", exec_space(), *local_grid, Cajita::Ghost(), Cajita::Cell(),
            KOKKOS_LAMBDA( const int i, const int j) {
                const auto kx = std::complex<double>(0.0, 2*M_PI/grid_points)
                    * static_cast<double>(i > grid_points/2 ? i - grid_points : 2*i == grid_points ? 0 : i);
                const auto ky = std::complex<double>(0.0, 2*M_PI/grid_points)
                    * static_cast<double>(j > grid_points/2 ? j - grid_points : 2*j == grid_points ? 0 : j);
                laplacian(i, j) = (kx*kx + ky*ky) * static_cast<double>(grid_points * grid_points) / (PFHUB_1A_SIZE*PFHUB_1A_SIZE);
            }
        );
    }

public:
    const int grid_points;
    const int timesteps;
    const double timestep_size;

    Simulation(CabanaPF::PFHubOne physics, int grid_points, int timesteps, double total_time)
        : grid_points{grid_points}, timesteps{timesteps}, physics{physics}, timestep_size{total_time/timesteps}
    {
        assert(NumSpaceDim==2); //this merge request just tests 2D
        //create global grid and mesh:
        auto global_mesh = Cajita::createUniformGlobalMesh(
            std::array<double, 2> {0, 0},
            std::array<double, 2> {PFHUB_1A_SIZE, PFHUB_1A_SIZE},  //TODO: Specify size in physics
            std::array<int, 2> {grid_points, grid_points}
        );
        Cajita::DimBlockPartitioner<2> partitioner;
        auto global_grid = Cajita::createGlobalGrid(MPI_COMM_WORLD, global_mesh, std::array<bool, 2>{true, true}, partitioner);

        //create local stuff:
        local_grid = Cajita::createLocalGrid(global_grid, 0);
        auto layout = createArrayLayout(local_grid, 2, Cajita::Cell()); //2: real & imag
        phis = PFVariables(layout, physics.variables());
        df_dphis = PFVariables(layout, physics.variables());
        //setup for FFT:
        fft = Cajita::Experimental::createHeffteFastFourierTransform<double, device_type>(*layout);
        setup_laplacian();
    }

    void initialize() {
        Cajita::grid_parallel_for("initialize", exec_space(), *local_grid, Cajita::Ghost(), Cajita::Cell(), physics.initializer(phis, 200/grid_points));
    }

    void timestep(int count) {
        assert(count+timesteps_done <= timesteps);
        for (int i=0; i<count; i++) {
            Cajita::grid_parallel_for("calculate df_dc", exec_space(), *local_grid, Cajita::Ghost(), Cajita::Cell(), physics.df_dphi(phis, df_dphis));
        }
        timesteps_done += count;
    }
};

}

#endif
