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
    std::unordered_map<std::string, int> name_lookup;
    std::array<int, NumSpaceDim> array_size;    //number of x, y, (and possibly z) points
public:
    std::array<CajitaArray, NumVariables> arrays;

    PFVariables(std::shared_ptr<Cajita::ArrayLayout<Cajita::Cell, Mesh>> layout, std::array<std::string, NumVariables> names) {
        //create an array and store the name of each variable:
        for(int i=0; i<NumVariables; i++) {
            arrays[i] = Cajita::createArray<double, device_type>(names[i], layout);
            name_lookup[names[i]] = i;
        }
        //Record the array size for each spatial dimension:
        const auto GlobalMesh = layout->localGrid()->globalGrid().globalMesh();
        for(int i=0; i<NumSpaceDim; i++) {
            array_size[i] = GlobalMesh.globalNumCell(i);
        }
    }

    View_type operator[](int index) {
        return arrays[index]->view();
    }

    //TODO: Load from config file?
    static inline const std::string SAVE_PATH = "/home/kokkos/src/CabanaPF/results/";
    //If unfinished and just saving progress, pass in timesteps_done
    void save(std::string problem_name, const int timesteps_done = -1) {
        for(int i=0; i<NumVariables; i++) {
            Cajita::Experimental::BovWriter::writeTimeStep(999999, 0, *arrays[i]);  //use 999999 to mark it for move
            try {
                std::string old_name = "grid_" + arrays[i]->label() + "_999999.dat";
                std::stringstream new_name;
                new_name << SAVE_PATH << problem_name << "_" << arrays[i]->label();
                if (timesteps_done > -1) {  //incomplete simulation, mark as such
                    new_name << ".tmp" << timesteps_done;
                }
                new_name << ".dat";
                std::filesystem::rename(old_name, new_name.str());
                //TODO: Deal with the .bov file?
            } catch (std::filesystem::filesystem_error &e) {
                std::cerr << "Error when saving: " << e.what() << std::endl;
            }
        }
    }
};

}

#endif
