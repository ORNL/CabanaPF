#include <iostream>
#include <cmath>
#include <Cabana_Core.hpp>
#include <Cajita.hpp>

const double PFHUB_1A_SIZE = 200.0;
const int GRID_POINTS = 96; //TODO: Runs with different mesh sizes
const int HALO_WIDTH = 2;

const double C0 = .5;
const double EPSILON = .01;

#define PFHUB_INITIAL(x, y) (C0 + EPSILON*(std::cos(.105*x)*std::cos(.11*y) \
    + std::pow(std::cos(.13*x)*std::cos(.087*y), 2) \
    + std::cos(.025*x-.15*y)*std::cos(.07*x-.02*y)))

class PfHubProblem {
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;
    using Mesh = Cajita::UniformMesh<double, 2>;
private:
    std::shared_ptr<Cajita::GlobalMesh<Mesh>> global_mesh;
    std::shared_ptr<Cajita::GlobalGrid<Mesh>> global_grid;
    std::shared_ptr<Cajita::LocalGrid<Mesh>> local_grid;
    std::shared_ptr<Cajita::Array<double, Cajita::Cell, Mesh, device_type>> data;
    Kokkos::View<double ***, device_type> data_view;
public:
    PfHubProblem() {
        //create global grid and mesh:
        global_mesh = Cajita::createUniformGlobalMesh(
            std::array<double, 2> {0, 0},
            std::array<double, 2> {PFHUB_1A_SIZE, PFHUB_1A_SIZE},  //a SIZE*SIZE square
            std::array<int, 2> {GRID_POINTS, GRID_POINTS}          //that has a GRID_POINTS*GRID_POINTS square mesh
        );
        Cajita::DimBlockPartitioner<2> partitioner;
        global_grid = Cajita::createGlobalGrid(MPI_COMM_WORLD, global_mesh, std::array<bool, 2>{true, true}, partitioner);
        //create local grid and array that holds all the data:
        local_grid = Cajita::createLocalGrid( global_grid, HALO_WIDTH );
        auto layout = createArrayLayout(global_grid, HALO_WIDTH, 4, Cajita::Cell());
            //4: c, dfdc, and their Fourier transforms
        std::string name( "concentrate" );
        data = Cajita::createArray<double, device_type>( name, layout );
        data_view = data->view();
    }

    //fills the grid with initial conditions as per PFHub benchmark 1
    void fill_initial() {
        Cajita::grid_parallel_for(
            "initialize", exec_space(), *local_grid, Cajita::Ghost(), Cajita::Cell(),
            KOKKOS_LAMBDA( const int i, const int j) {
                const double x = i*PFHUB_1A_SIZE/GRID_POINTS;
                const double y = j*PFHUB_1A_SIZE/GRID_POINTS;
                data_view(i, j, 0) = PFHUB_INITIAL(x, y);
            }
        );
    }

    //Returns the current concentrate at a given grid point
    double get_c(int i, int j) {
        return data_view(i, j, 0);
    }
};

#ifndef BUILD_FOR_TESTS
int main( int argc, char* argv[] ) {
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );
        PfHubProblem simulation;
        simulation.fill_initial();
        for(int i=0; i<10; i++) {
            std::cout << simulation.get_c(i, 0) << std::endl;
        }
    }
    MPI_Finalize();
    std::cout << "done" << std::endl;
    return 0;
}
#endif
