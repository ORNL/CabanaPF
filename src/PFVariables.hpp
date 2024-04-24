#ifndef PFVARIABLES_H
#define PFVARIABLES_H

#include <Cabana_Grid.hpp>
#include <fstream>

namespace CabanaPF {

template <std::size_t NumSpaceDim, std::size_t NumVariables>
class PFVariables {
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename execution_space::memory_space;
    using Mesh = Cabana::Grid::UniformMesh<double, NumSpaceDim>;
    using GridArray = std::shared_ptr<Cabana::Grid::Array<double, Cabana::Grid::Node, Mesh, memory_space>>;
    using View_type = std::conditional_t<3 == NumSpaceDim, Kokkos::View<double****, memory_space>,
                                         Kokkos::View<double***, memory_space>>;

  private:
    std::array<int, NumSpaceDim> array_size; // number of x, y, (and possibly z) points
    std::shared_ptr<Cabana::Grid::Experimental::HeffteFastFourierTransform<
        Cabana::Grid::Node, Mesh, double, memory_space, execution_space,
        Cabana::Grid::Experimental::Impl::FFTBackendDefault>>
        fft_calculator;

    // string to be used for Cabana's BOVWriter's prefix argument
    std::string save_prefix(const int index, std::string run_name) {
        std::stringstream name;
#ifdef CABANAPF_RESULTS_PATH
        name << CABANAPF_RESULTS_PATH;
#endif
        name << run_name << "_" << arrays[index]->label();
        return name.str();
    }

  public:
    std::array<GridArray, NumVariables> arrays;

    PFVariables(std::shared_ptr<Cabana::Grid::ArrayLayout<Cabana::Grid::Node, Mesh>> layout,
                std::array<std::string, NumVariables> names) {
        // create an array and store the name of each variable:
        for (std::size_t i = 0; i < NumVariables; i++) {
            arrays[i] = Cabana::Grid::createArray<double, memory_space>(names[i], layout);
        }
        // Record the array size for each spatial dimension:
        const auto GlobalMesh = layout->localGrid()->globalGrid().globalMesh();
        for (std::size_t i = 0; i < NumSpaceDim; i++) {
            array_size[i] = GlobalMesh.globalNumCell(i);
        }
        fft_calculator = Cabana::Grid::Experimental::createHeffteFastFourierTransform<double, memory_space>(*layout);
    }

    void fft_forward(int index) {
        fft_calculator->forward(*arrays[index], Cabana::Grid::Experimental::FFTScaleNone());
    }

    void fft_inverse(int index) {
        fft_calculator->reverse(*arrays[index], Cabana::Grid::Experimental::FFTScaleFull());
    }

    View_type operator[](int index) {
        return arrays[index]->view();
    }

    /*If on GPU, will copy over the data to CPU and return it
    If on CPU, same as using []*/
    auto host_view(int index) {
        return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), arrays[index]->view());
    }

    // Saves an array to file.
    void save(const int index, std::string run_name, const int timesteps_done, const double time = 0) {
        const std::string prefix = save_prefix(index, run_name);
        Cabana::Grid::Experimental::BovWriter::writeTimeStep(prefix, timesteps_done, time, *arrays[index]);
    }

    // loads the file created by the same arguments to save
    void load(const int index, std::string run_name, const int timesteps_done) {
        assert(NumSpaceDim == 2); // currently not supported for 3D
        // recreate Cabana's naming scheme:
        std::stringstream name;
        name << save_prefix(index, run_name) << "_" << std::setfill('0') << std::setw(6) << timesteps_done << ".dat";
        // open the file:
        std::fstream infile(name.str(), std::fstream::in | std::fstream::binary);
        // read it:
        const auto view = arrays[index]->view();
        double buffer[2];
        for (int j = 0; j < array_size[1]; j++) {
            for (int i = 0; i < array_size[0]; i++) {
                infile.read((char*)buffer, 2 * sizeof(double));
                view(i, j, 0) = buffer[0];
                view(i, j, 1) = buffer[1];
            }
            // since the grid is periodic, it writes the first value in the row again, so "burn it off":
            infile.read((char*)buffer, 2 * sizeof(double));
        }
    }
};

} // namespace CabanaPF

#endif
