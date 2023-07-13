#ifndef PFVARIABLES_H
#define PFVARIABLES_H

#include <Cajita.hpp>
#include <fstream>
#include <filesystem>

namespace CabanaPF {

template <std::size_t NumSpaceDim, std::size_t NumVariables>
class PFVariables {
    using device_type = Kokkos::DefaultExecutionSpace::device_type;
    using Mesh = Cajita::UniformMesh<double, NumSpaceDim>;
    using CajitaArray = std::shared_ptr<Cajita::Array<double, Cajita::Cell, Mesh, device_type>>;
    using View_type = std::conditional_t<3 == NumSpaceDim, Kokkos::View<double****>, Kokkos::View<double***>>;
private:
    std::array<int, NumSpaceDim> array_size;    //number of x, y, (and possibly z) points
    std::shared_ptr<Cajita::Experimental::HeffteFastFourierTransform<
        Cajita::Cell, Mesh, double, device_type, Cajita::Experimental::Impl::FFTBackendDefault>> fft_calculator;
public:
    std::array<CajitaArray, NumVariables> arrays;

    PFVariables(std::shared_ptr<Cajita::ArrayLayout<Cajita::Cell, Mesh>> layout, std::array<std::string, NumVariables> names) {
        //create an array and store the name of each variable:
        for(int i=0; i<NumVariables; i++) {
            arrays[i] = Cajita::createArray<double, device_type>(names[i], layout);
        }
        //Record the array size for each spatial dimension:
        const auto GlobalMesh = layout->localGrid()->globalGrid().globalMesh();
        for(int i=0; i<NumSpaceDim; i++) {
            array_size[i] = GlobalMesh.globalNumCell(i);
        }
        fft_calculator = Cajita::Experimental::createHeffteFastFourierTransform<double, device_type>(*layout);
    }

    void fft(int index) {
        fft_calculator->forward(*arrays[index], Cajita::Experimental::FFTScaleNone());
    }

    void ifft(int index) {
        fft_calculator->reverse(*arrays[index], Cajita::Experimental::FFTScaleFull());
    }

    //TODO: Remove
    PFVariables(CajitaArray arr) {
        arrays[0] = arr;
    }

    View_type operator[](int index) {
        return arrays[index]->view();
    }

    //TODO: Load from config file?
    static inline const std::string SAVE_PATH = "/home/kokkos/src/CabanaPF/results/";
    std::string save_name(std::string run_name, const int index, const int timesteps_done = -1) {
        std::stringstream name;
        name << SAVE_PATH << run_name << "_" << arrays[index]->label();
        if (timesteps_done > -1) {  //incomplete simulation, mark as such
            name << ".tmp" << timesteps_done;
        }
        name << ".dat";
        return name.str();
    }

    //If unfinished and just saving progress, pass in timesteps_done
    void save(const int index, std::string run_name, const int timesteps_done = -1) {
        Cajita::Experimental::BovWriter::writeTimeStep(999999, 0, *arrays[index]);  //use 999999 to mark it for move
        try {
            std::string old_name = "grid_" + arrays[index]->label() + "_999999.dat";
            std::filesystem::rename(old_name, save_name(run_name, index, timesteps_done));
            //TODO: Deal with the .bov file?
        } catch (std::filesystem::filesystem_error &e) {
            std::cerr << "Error when saving: " << e.what() << std::endl;
        }
    }

    void load(std::string run_name, const int timesteps_done = -1) {
        for (int index=0; index<NumVariables; index++) {
            assert(NumSpaceDim==2);    //currently not supported for 3D
            //open the file:
            std::fstream infile(save_name(run_name, index, timesteps_done), std::fstream::in|std::fstream::binary);
            //read it:
            const auto view = arrays[index]->view();
            double buffer[2];
            //for (int k=0; k < (NumSpaceDim<3 ? 1 : array_size[2]); k++) {
                for (int j=0; j<array_size[1]; j++) {
                    for (int i=0; i<array_size[0]; i++) {
                        infile.read((char*)buffer, 2*sizeof(double));
                        //loadHelper(view, i, j, k, buffer);
                        view(i, j, 0) = buffer[0];
                        view(i, j, 1) = buffer[1];
                    }
                }
            //}
        }
    }
};

/*Not sure why this doesn't work
template<std::size_t NumVariables>
class PFVariables<2, NumVariables> {
private:
    void loadHelper(Kokkos::View<double***> &view, int i, int j, int k, double *buffer) {
        view(i, j, 0) = buffer[0];
        view(i, j, 1) = buffer[1];
    }
};
template<std::size_t NumVariables>
class PFVariables<3, NumVariables> {
private:
    void loadHelper(Kokkos::View<double****> &view, int i, int j, int k, double *buffer) {
        view(i, j, k, 0) = buffer[0];
        view(i, j, k, 1) = buffer[1];
    }
};
*/
}

#endif
