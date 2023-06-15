#ifndef PFHUB_H
#define PFHUB_H

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <cmath>

namespace CabanaPF
{
const double PFHUB_1A_SIZE = 200.0;
const double C0 = .5;
const double EPSILON = .01;

//This will probably get moved into an "Initializer" class once we have more problems
double PFHUB_INITIAL(double x, double y) {
    return C0 + EPSILON*(std::cos(.105*x)*std::cos(.11*y)
    + std::pow(std::cos(.13*x)*std::cos(.087*y), 2)
    + std::cos(.025*x-.15*y)*std::cos(.07*x-.02*y));
}

class PfHubProblem {
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;
    using Mesh = Cajita::UniformMesh<double, 2>;
private:
    int grid_points;
    std::shared_ptr<Cajita::LocalGrid<Mesh>> local_grid;
    std::shared_ptr<Cajita::Array<double, Cajita::Cell, Mesh, device_type>> data;
    Kokkos::View<double ***, device_type> data_view;
public:
    PfHubProblem(int grid_points) { //number of points used in each dimension
        this->grid_points = grid_points;
        //create global grid and mesh:
        auto global_mesh = Cajita::createUniformGlobalMesh(
            std::array<double, 2> {0, 0},
            std::array<double, 2> {PFHUB_1A_SIZE, PFHUB_1A_SIZE},  //a SIZE*SIZE square
            std::array<int, 2> {grid_points, grid_points}          //that has a GRID_POINTS*GRID_POINTS square mesh
        );
        Cajita::DimBlockPartitioner<2> partitioner;
        auto global_grid = Cajita::createGlobalGrid(MPI_COMM_WORLD, global_mesh, std::array<bool, 2>{true, true}, partitioner);
        //create local grid and array that holds all the data:
        local_grid = Cajita::createLocalGrid( global_grid, 0 );
        auto layout = createArrayLayout(global_grid, 0, 4, Cajita::Cell());
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
                const double x = i*PFHUB_1A_SIZE/grid_points;
                const double y = j*PFHUB_1A_SIZE/grid_points;
                data_view(i, j, 0) = PFHUB_INITIAL(x, y);
            }
        );
    }

    //Returns the current concentration at a given grid point
    double get_c(int i, int j) {
        return data_view(i, j, 0);
    }
};
}

#endif