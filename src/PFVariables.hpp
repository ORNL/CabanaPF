#ifndef PFVARIABLES_H
#define PFVARIABLES_H

#include <Cabana_Grid.hpp>
#include <fstream>

#ifdef RESULTS_PATH
#include <filesystem>
#endif

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

    std::string save_name(std::string run_name, [[maybe_unused]] const int index,
                          [[maybe_unused]] const int timesteps_done = -1) {
        std::stringstream name;
#ifdef RESULTS_PATH
        name << RESULTS_PATH;
#endif
        name << run_name << "_" << arrays[index]->label() << ".dat";
        return name.str();
    }

    // If unfinished and just saving progress, pass in timesteps_done
    void save(const int index, [[maybe_unused]] std::string run_name, [[maybe_unused]] const int timesteps_done = -1) {
        Cabana::Grid::Experimental::BovWriter::writeTimeStep(999999, 0,
                                                             *arrays[index]); // use 999999 to mark it for move
#ifdef RESULTS_PATH // Comes from the CMake build; if not defined, won't have file I/O
        try {
            std::string old_name = "grid_" + arrays[index]->label() + "_999999.dat";
            std::filesystem::rename(old_name, save_name(run_name, index, timesteps_done));
            // TODO: Deal with the .bov file?
        } catch (std::filesystem::filesystem_error& e) {
            std::cerr << "Error when saving: " << e.what() << std::endl;
        }
#endif
    }

    void load(std::string run_name, const int timesteps_done = -1) {
        for (std::size_t index = 0; index < NumVariables; index++) {
            assert(NumSpaceDim == 2); // currently not supported for 3D
            // open the file:
            std::fstream infile(save_name(run_name, index, timesteps_done), std::fstream::in | std::fstream::binary);
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
    }
};

} // namespace CabanaPF

#endif
