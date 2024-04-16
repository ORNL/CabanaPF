#ifndef PFHUB_H
#define PFHUB_H

#include <Cabana_Grid.hpp>
#include <PFVariables.hpp>
#include <Runner.hpp>

namespace CabanaPF {

/*The PFHub Benchmark 1a: Spinodal Decomposition (https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/).  We have
  two versions:
     -PFHub1aBenchmark (which uses the actual benchmark initial conditions)
     -PFHub1aPeriodic (which uses similar but periodic initial conditions)
*/
class PFHub1aBase : public CabanaPFRunner<2> {
  protected:
    using cdouble = Kokkos::complex<double>;

    Kokkos::View<cdouble**, device_type> laplacian_view;
    PFVariables<2, 2> vars;

  public:
    static constexpr double _SIZE = 200.;
    static constexpr double _KAPPA = 2.;
    static constexpr double _M = 5.;
    static constexpr double _RHO = 5.;
    static constexpr double _C_ALPHA = .3;
    static constexpr double _C_BETA = .7;

    const int grid_points;
    const double cell_size;
    const double dt;

    PFHub1aBase(int grid_points, int timesteps_per_t)
        : CabanaPFRunner(grid_points, _SIZE, timesteps_per_t), vars{layout, {"c", "df_dc"}},
          grid_points{grid_points}, cell_size{_SIZE / grid_points}, dt{1.0 / timesteps_per_t} {
        laplacian_view = Kokkos::View<cdouble**, device_type>("laplacian", grid_points, grid_points);
    }

    double get_c(int i, int j) {
        return vars[0](i, j, 0);
    }

    auto get_cpu_view() {
        return vars.host_view(0);
    }

    // Problem-specific initial conditions
    virtual void initial_conditions() = 0;

    void initialize() override {
        // setup the laplacian:
        const auto laplacian = laplacian_view;
        const auto points = grid_points;
        const double SIZE = _SIZE;

        node_parallel_for(
            "laplacian", KOKKOS_LAMBDA(const int i, const int j) {
                const auto kx = cdouble(0.0, 2 * M_PI / points) * static_cast<double>(i > points / 2    ? i - points
                                                                                      : 2 * i == points ? 0
                                                                                                        : i);
                const auto ky = cdouble(0.0, 2 * M_PI / points) * static_cast<double>(j > points / 2    ? j - points
                                                                                      : 2 * j == points ? 0
                                                                                                        : j);
                laplacian(i, j) = (kx * kx + ky * ky) * static_cast<double>(points * points) / (SIZE * SIZE);
            });
        // have the problem do its setup:
        initial_conditions();
    }

    void calc_dfdc() {
        // Calculate df_dc values:
        const auto c_view = vars[0];
        const auto dfdc_view = vars[1];
        const double RHO = _RHO, C_ALPHA = _C_ALPHA, C_BETA = _C_BETA;

        node_parallel_for(
            "df_dc", KOKKOS_LAMBDA(const int i, const int j) {
                const cdouble c(c_view(i, j, 0), c_view(i, j, 1));
                const cdouble df_dc = RHO * (2.0 * (c - C_ALPHA) * (C_BETA - c) * (C_BETA - c) -
                                             2.0 * (C_BETA - c) * (c - C_ALPHA) * (c - C_ALPHA));
                dfdc_view(i, j, 0) = df_dc.real();
                dfdc_view(i, j, 1) = df_dc.imag();
            });
    }

    void step() override {
        calc_dfdc();
        // enter Fourier space:
        vars.fft_forward(0);
        vars.fft_forward(1);

        // step forward with semi-implicit Euler in Fourier space:
        const double dt = this->dt, M = _M, KAPPA = _KAPPA;
        const auto c = vars[0];
        const auto df_dc = vars[1];
        const auto laplacian = laplacian_view;
        node_parallel_for(
            "timestep", KOKKOS_LAMBDA(const int i, const int j) {
                const cdouble df_dc_hat(df_dc(i, j, 0), df_dc(i, j, 1));
                cdouble c_hat(c(i, j, 0), c(i, j, 1));

                c_hat = (c_hat + dt * M * laplacian(i, j) * df_dc_hat) /
                        (1.0 + dt * M * KAPPA * laplacian(i, j) * laplacian(i, j));
                c(i, j, 0) = c_hat.real();
                c(i, j, 1) = c_hat.imag();
            });
        // rescue concentration values from Fourier space (dfdc can be left there since it gets recalculated next
        // timestep anyways)
        vars.fft_inverse(0);
    }

    virtual void output() {}

    // needed to allow polymorphism:
    virtual ~PFHub1aBase() {}
};

class PFHub1aBenchmark : public PFHub1aBase {
  public:
    void initial_conditions() override {
        const auto c = vars[0]; // get View for scope capture
        const auto delta = cell_size;
        node_parallel_for(
            "benchmark initial conditions", KOKKOS_LAMBDA(const int i, const int j) {
                // initialize c:
                const double x = delta * i;
                const double y = delta * j;
                c(i, j, 0) = .5 + .01 * (Kokkos::cos(.105 * x) * Kokkos::cos(.11 * y) +
                                         Kokkos::cos(.13 * x) * Kokkos::cos(.087 * y) * Kokkos::cos(.13 * x) *
                                             Kokkos::cos(.087 * y) +
                                         Kokkos::cos(.025 * x - .15 * y) * Kokkos::cos(.07 * x - .02 * y));
                c(i, j, 1) = 0;
            });
    }

    void output() override {
        std::stringstream s;
        s << "1aBenchmark_N" << grid_points << "TS" << timesteps_done;
        vars.save(0, s.str());
    }

    PFHub1aBenchmark(int grid_points, int timesteps_per_t) : PFHub1aBase{grid_points, timesteps_per_t} {}
};

class PFHub1aPeriodic : public PFHub1aBase {
  public:
    void initial_conditions() override {
        const auto c = vars[0]; // get View for scope capture
        const auto delta = cell_size;
        node_parallel_for(
            "periodic initial conditions", KOKKOS_LAMBDA(const int i, const int j) {
                // initialize c:
                const double x = delta * i;
                const double y = delta * j;
                c(i, j, 0) = .5 + .01 * (Kokkos::cos(3 * M_PI * x / 100) * Kokkos::cos(M_PI * y / 25) +
                                         Kokkos::cos(M_PI * x / 25) * Kokkos::cos(3 * M_PI * y / 100) *
                                             Kokkos::cos(M_PI * x / 25) * Kokkos::cos(3 * M_PI * y / 100) +
                                         Kokkos::cos(M_PI * x / 100 - M_PI * y / 20) *
                                             Kokkos::cos(M_PI * x / 50 - M_PI * y / 100));
                c(i, j, 1) = 0;
            });
    }

    void output() override {
        std::stringstream s;
        s << "1aPeriodic_N" << grid_points << "TS" << timesteps_per_t;
        vars.save(0, s.str());
    }

    PFHub1aPeriodic(int grid_points, int timesteps) : PFHub1aBase{grid_points, timesteps} {}
};

} // namespace CabanaPF

#endif
