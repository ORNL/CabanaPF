#ifndef PFVARIABLES_H
#define PFVARIABLES_H

#include <Cajita.hpp>
#include <fstream>

namespace CabanaPF {

template <std::size_t NumSpaceDim, std::size_t NumVariables>
class PFVariables {
    using device_type = Kokkos::DefaultExecutionSpace::device_type;
    using Mesh = Cajita::UniformMesh<double, NumSpaceDim>;
    using CajitaArray = std::shared_ptr<Cajita::Array<double, Cajita::Cell, Mesh, device_type>>;
    using View_type = std::conditional_t<3 == NumSpaceDim, Kokkos::View<double****>, Kokkos::View<double***>>;
private:
    std::unordered_map<std::string, int> name_lookup;
    std::array<int, NumSpaceDim> array_size;
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

    void save(const int index, const int timestep) {
        Cajita::Experimental::BovWriter::writeTimeStep(timestep, 0, *arrays[index]);
    }
};

}

#endif
