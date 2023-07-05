#ifndef PFHUB_H
#define PFHUB_H

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <cmath>
#include <complex>

#include <PFVariables.hpp>

namespace CabanaPF
{
const double PFHUB_1A_SIZE = 200.0;
const double C0 = .5;
const double EPSILON = .01;
const double RHO = 5.0;
const double M = 5.0;
const double KAPPA = 2.0;
const double C_ALPHA = .3;
const double C_BETA = .7;

//This will probably get moved into an "Initializer" class once we have more problems
/*double PFHUB_INITIAL(double x, double y) {
    return C0 + EPSILON*(std::cos(.105*x)*std::cos(.11*y)
    + std::pow(std::cos(.13*x)*std::cos(.087*y), 2)
    + std::cos(.025*x-.15*y)*std::cos(.07*x-.02*y));
}*/

double PFHUB_INITIAL(double x, double y) {
    return C0 + EPSILON*(std::cos(M_PI*x/50)*std::cos(M_PI*y/100));
}

class PfHubProblem {
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;
    using Mesh = Cajita::UniformMesh<double, 2>;
    using cdouble = std::complex<double>;
private:
    int grid_points;
    std::shared_ptr<Cajita::LocalGrid<Mesh>> local_grid;
    std::shared_ptr<Cajita::Experimental::HeffteFastFourierTransform<
        Cajita::Cell, Mesh, double, device_type, Cajita::Experimental::Impl::FFTBackendDefault>> fft;

    std::shared_ptr<Cajita::Array<double, Cajita::Cell, Mesh, device_type>> concentration;
    std::shared_ptr<Cajita::Array<double, Cajita::Cell, Mesh, device_type>> dfdc;

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
        //create local grid and arrays for c and dfdc:
        local_grid = Cajita::createLocalGrid( global_grid, 0 );
        auto layout = createArrayLayout(local_grid, 2, Cajita::Cell());
        concentration = Cajita::createArray<double, device_type>("concentration", layout);
        dfdc = Cajita::createArray<double, device_type>("dfdc", layout);
        //setup for FFT:
        fft = Cajita::Experimental::createHeffteFastFourierTransform<double, device_type>(*layout);
        setup_laplacian();
    }

    //fills the grid with initial conditions as per PFHub benchmark 1
    void fill_initial() {
        Cajita::grid_parallel_for(
            "initialize", exec_space(), *local_grid, Cajita::Ghost(), Cajita::Cell(),
            KOKKOS_LAMBDA( const int i, const int j) {
                const double x = i*PFHUB_1A_SIZE/grid_points;
                const double y = j*PFHUB_1A_SIZE/grid_points;
                concentration->view()(i, j, 0) = PFHUB_INITIAL(x, y);
                concentration->view()(i, j, 1) = 0;
            }
        );
    }

    //Returns the current concentration at a given grid point
    double get_c(int i, int j) {
        return concentration->view()(i, j, 0);
    }

    //advances the problem using pseudospectral methods
    void timestep(double dt, int timesteps) {
        for(int t=0; t<timesteps; t++) {
            //calculate df_dc:
            Cajita::grid_parallel_for(
                "calculate df_dc", exec_space(), *local_grid, Cajita::Ghost(), Cajita::Cell(),
                KOKKOS_LAMBDA( const int i, const int j) {
                    const cdouble c(concentration->view()(i, j, 0), concentration->view()(i, j, 1));
                    const cdouble df_dc = RHO * (2.0*(c-C_ALPHA)*(C_BETA-c)*(C_BETA-c) - 2.0*(C_BETA-c)*(c-C_ALPHA)*(c-C_ALPHA));
                    dfdc->view()(i, j, 0) = df_dc.real();
                    dfdc->view()(i, j, 1) = df_dc.imag();
                }
            );
            //enter Fourier space:
            fft->forward(*concentration, Cajita::Experimental::FFTScaleNone());
            fft->forward(*dfdc, Cajita::Experimental::FFTScaleNone());
            //step in Fourier space:
            Cajita::grid_parallel_for(
                "forward step", exec_space(), *local_grid, Cajita::Ghost(), Cajita::Cell(),
                KOKKOS_LAMBDA(const int i, const int j) {
                    const cdouble df_dc_hat(dfdc->view()(i, j, 0), dfdc->view()(i, j, 1));
                    cdouble c_hat(concentration->view()(i, j, 0), concentration->view()(i, j, 1));

                    c_hat = (c_hat + dt*M*laplacian(i, j)*df_dc_hat) / (1.0 + dt*M*KAPPA*laplacian(i, j)*laplacian(i, j));
                    concentration->view()(i, j, 0) = c_hat.real();
                    concentration->view()(i, j, 1) = c_hat.imag();
                }
            );
            //leave Fourier space:
            fft->reverse(*concentration, Cajita::Experimental::FFTScaleFull());
        }
        CabanaPF::PFVariables<2,1> pfv(concentration);
        std::stringstream ss;
        ss << "PFHub1a_N" << grid_points << "T" << timesteps;
        pfv.save(ss.str());
    }
};
}
#endif
