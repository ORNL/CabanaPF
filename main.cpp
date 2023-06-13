#include <iostream>
#include <cmath>
#include <Cabana_Core.hpp>
#include <Cajita.hpp>

const double PFHUB_1A_SIZE = 200.0;
const int GRID_POINTS = 96; //TODO: Runs with different mesh sizes

const double C0 = .5;
const double EPSILON = .01;

#define PFHUB_INITIAL(x, y) (C0 + EPSILON*(std::cos(.105*x)*std::cos(.11*y) \
    + std::pow(std::cos(.13*x)*std::cos(.087*y), 2) \
    + std::cos(.025*x-.15*y)*std::cos(.07*x-.02*y)))

//fill the array with the specified initial conditions (PFHub benchmark 1a)
template <class ExecutionSpace, class LocalGridType, class ArrayType>
void fill_initial(ExecutionSpace exec_space, LocalGridType local_grid, ArrayType& field) {
    Cajita::grid_parallel_for(
        "initialize", exec_space, *local_grid, Cajita::Ghost(), Cajita::Cell(),
        KOKKOS_LAMBDA( const int i, const int j) {
            const double x = i*PFHUB_1A_SIZE/GRID_POINTS;
            const double y = j*PFHUB_1A_SIZE/GRID_POINTS;
            field(i, j, 0) = PFHUB_INITIAL(x, y)
        }
    );
}

//currently this is just everything
void setup() {
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;
    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    //create mesh and grid:
    auto global_mesh = Cajita::createUniformGlobalMesh(
        std::array<double, 2> {0, 0},
        std::array<double, 2> {PFHUB_1A_SIZE, PFHUB_1A_SIZE},  //a SIZE*SIZE square
        std::array<int, 2> {GRID_POINTS, GRID_POINTS}          //that has a GRID_POINTS*GRID_POINTS square mesh
    );
    Cajita::DimBlockPartitioner<2> partitioner;
    auto global_grid = Cajita::createGlobalGrid(MPI_COMM_WORLD, global_mesh, std::array<bool, 2>{true, true}, partitioner);

    //create array to store all the data:
    int halo_width = 2;
    auto local_grid = Cajita::createLocalGrid( global_grid, halo_width );
    auto layout = createArrayLayout(global_grid, halo_width, 4, Cajita::Cell());
        //4: c, dfdc, and their Fourier transforms
    std::string name( "concentrate" );
    auto concentrate = Cajita::createArray<double, device_type>( name, layout );
    auto conc_view = concentrate->view();


    fill_initial(exec_space(), local_grid, conc_view);
}

int main( int argc, char* argv[] ) {
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );
        setup();
    }
    MPI_Finalize();
    std::cout << "done" << std::endl;
    return 0;
}
