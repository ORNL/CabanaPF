#ifndef PHYSICS_H
#define PHYSICS_H

#include <vector>
#include <string>
#include <complex>

#include <PFVariables.hpp>

namespace CabanaPF {

//Inherit from this to implement the problem-specific physics
template<std::size_t NumSpaceDim, std::size_t NumVariables>
class Physics {
public:
    using cdouble = std::complex<double>;
    virtual std::array<std::string, NumVariables> variables() = 0;
    virtual std::function<void(int, int)> initializer(PFVariables<NumSpaceDim, NumVariables> vars, double cell_size) = 0;
    virtual std::function<void(int, int)> df_dphi(PFVariables<NumSpaceDim, NumVariables> vars,
        PFVariables<NumSpaceDim, NumVariables> partials) = 0;
};

class PFHubOne : public Physics<2, 1> {
public:
    std::array<std::string, 1> variables() {
        return {"c"};
    }
    static constexpr double C0 = .5;
    static constexpr double EPSILON = .01;
    const double RHO = 5.0;
    const double M = 5.0;
    const double KAPPA = 2.0;
    const double C_ALPHA = .3;
    const double C_BETA = .7;

    std::function<void(int, int)> initializer(PFVariables<2, 1> vars, double cell_size) {
        return KOKKOS_LAMBDA( const int i, const int j) {
            const double x = i*cell_size, y = j*cell_size;
            vars.arrays[0]->view()(i, j, 0) = C0 + EPSILON*(std::cos(.105*x)*std::cos(.11*y)
                + std::pow(std::cos(.13*x)*std::cos(.087*y), 2)
                + std::cos(.025*x-.15*y)*std::cos(.07*x-.02*y));
            vars.arrays[0]->view()(i, j, 1) = 0;
        };
    }

    std::function<void(int, int)> df_dphi(PFVariables<2, 1> vars, PFVariables<2, 1> partials) {
        return KOKKOS_LAMBDA(const int i, const int j) {
            const auto c_view = vars.arrays[0]->view();
            const cdouble c(c_view(i, j, 0), c_view(i, j, 1));
            const cdouble df_dc = 2 * RHO * (c-C_ALPHA) * (c-C_BETA) * (2.*c-C_ALPHA-C_BETA);
        };
    }
};

}

#endif
