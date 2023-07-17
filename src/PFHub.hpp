#ifndef PFHUB_H
#define PFHUB_H

#include <Cajita.hpp>
#include <Runner.hpp>
#include <PFVariables.hpp>

namespace CabanaPF {

/*The PFHub Benchmark 1a: Spinodal Decomposition (https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/).  We have two versions:
    -PFHub1aBenchmark (which uses the actual benchmark initial conditions)
    -PFHub1aPeriodic (which uses similar but periodic initial conditions)
*/
class PFHub1aBase : public CabanaPFRunner<2> {
protected:
    using cdouble = Kokkos::complex<double>;

    const double cell_size;
    const int timesteps;
    const int grid_points;
    Kokkos::View<cdouble **, device_type> laplacian_view;

public:
    PFVariables<2, 2> vars;
    static constexpr double SIZE = 200.;

    PFHub1aBase(int grid_points, int timesteps) : CabanaPFRunner(grid_points, timesteps, SIZE),
        vars{layout, {"c", "df_dc"}}, cell_size{SIZE/grid_points}, timesteps{timesteps}, grid_points{grid_points}
    {
        laplacian_view = Kokkos::View<cdouble **> ("laplacian", grid_points, grid_points);
    }

    double get_c(int i, int j) {
        return vars[0](i, j, 0);
    }

    //Problem-specific initial conditions
    virtual void initial_conditions()=0;
    
    void initialize() override {
        //setup the laplacian:
        const auto laplacian = laplacian_view;
        CABANAPF_PARALLEL("laplacian", KOKKOS_LAMBDA(const int i, const int j) {
            const auto kx = cdouble(0.0, 2*M_PI/grid_points)
                    * static_cast<double>(i > grid_points/2 ? i - grid_points : 2*i == grid_points ? 0 : i);
            const auto ky = cdouble(0.0, 2*M_PI/grid_points)
                * static_cast<double>(j > grid_points/2 ? j - grid_points : 2*j == grid_points ? 0 : j);
            laplacian(i, j) = (kx*kx + ky*ky) * static_cast<double>(grid_points * grid_points) / (SIZE*SIZE);
        });
        //have the problem to its setup:
        initial_conditions();
    }

    void pre_step() override {
        //Calculate df_dc values:
        const auto c_view = vars[0];
        const auto dfdc_view = vars[1];
        CABANAPF_PARALLEL("df_dc", KOKKOS_LAMBDA( const int i, const int j) {
            const cdouble c(c_view(i, j, 0), c_view(i, j, 1));
            const cdouble df_dc = 5 * (2.0*(c-.3)*(.7-c)*(.7-c) - 2.0*(.7-c)*(c-.3)*(c-.3));
            dfdc_view(i, j, 0) = df_dc.real();
            dfdc_view(i, j, 1) = df_dc.imag();
        });
    }

    void step() override {
        //enter Fourier space:
        vars.fft(0);
        vars.fft(1);
        
        const double dt = 250./timesteps;
        const auto c = vars[0];
        const auto df_dc = vars[1];
        const auto laplacian = laplacian_view;
        CABANAPF_PARALLEL("timestep", KOKKOS_LAMBDA(const int i, const int j) {
            const cdouble df_dc_hat(df_dc(i, j, 0), df_dc(i, j, 1));
            cdouble c_hat(c(i, j, 0), c(i, j, 1));

            c_hat = (c_hat + dt*5*laplacian(i, j)*df_dc_hat) / (1.0 + dt*5*2*laplacian(i, j)*laplacian(i, j));
            c(i, j, 0) = c_hat.real();
            c(i, j, 1) = c_hat.imag();
        });
    }

    void post_step() override {
        //rescue concentration values from Fourier space:
        vars.ifft(0);
    }
};

class PFHub1aBenchmark : public PFHub1aBase {
public:
    void initial_conditions() override {
        const auto c = vars[0];   //get View for scope capture
        CABANAPF_PARALLEL("benchmark initial conditions", KOKKOS_LAMBDA(const int i, const int j) {
            //initialize c:
            const double x = cell_size*i;
            const double y = cell_size*j;
            c(i, j, 0) = .5 + .01*(Kokkos::cos(.105*x)*Kokkos::cos(.11*y)
                + Kokkos::cos(.13*x)*Kokkos::cos(.087*y)*Kokkos::cos(.13*x)*Kokkos::cos(.087*y)
                + Kokkos::cos(.025*x-.15*y)*Kokkos::cos(.07*x-.02*y));
            c(i, j, 1) = 0;
        });
    }

    void finalize() override {
        std::stringstream s;
        s << "1aBenchmark_N" << grid_points << "T" << timesteps;
        vars.save(0, s.str());
    }

    PFHub1aBenchmark(int grid_points, int timesteps) : PFHub1aBase{grid_points, timesteps} {}
};

class PFHub1aPeriodic : public PFHub1aBase {
public:
    void initial_conditions() override {
        const auto c = vars[0];   //get View for scope capture
        CABANAPF_PARALLEL("periodic initial conditions", KOKKOS_LAMBDA(const int i, const int j) {
            //initialize c:
            const double x = cell_size*i;
            const double y = cell_size*j;
            c(i, j, 0) = .5 + .01*(Kokkos::cos(3*M_PI*x/100)*Kokkos::cos(M_PI*y/25)
                + Kokkos::cos(M_PI*x/25)*Kokkos::cos(3*M_PI*y/100)*Kokkos::cos(M_PI*x/25)*Kokkos::cos(3*M_PI*y/100)
                + Kokkos::cos(M_PI*x/100-M_PI*y/20)*Kokkos::cos(M_PI*x/50-M_PI*y/100));
            c(i, j, 1) = 0;
        });
    }

    void finalize() override {
        std::stringstream s;
        s << "1aPeriodic_N" << grid_points << "T" << timesteps;
        vars.save(0, s.str());
    }

    PFHub1aPeriodic(int grid_points, int timesteps) : PFHub1aBase{grid_points, timesteps} {}
};

}

#endif
