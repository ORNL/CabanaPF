#ifndef RUNNER_H
#define RUNNER_H

namespace CabanaPF {

//Inherit from this to implement the problem's specific actions
template <std::size_t NumSpaceDim, std::size_t NumVars>
class IRunner {
public:
    using KokkosFunc = std::conditional_t<3 == NumSpaceDim, std::function<void(int, int, int)>, std::function<void(int, int)>>;
    //mandatory things a user must implement:
    virtual KokkosFunc initialize() = 0;  //run once, when starting a new simulation
    virtual KokkosFunc step() = 0;        //run each timestep

    //optional things.  If unimplemented, Simulation will catch the nullptr and not dispatch to Kokkos
    KokkosFunc pre_step() {
        return nullptr;
    };
    KokkosFunc post_step() {
        return nullptr;
    };
    //also optional; called when the required number of timesteps have been done
    void finalize() {}
};

}

#endif
