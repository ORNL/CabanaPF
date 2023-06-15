#include <gtest/gtest.h>
#include <Cabana_Core.hpp>
#include <Cajita.hpp>

#include <PFHub.hpp>

using namespace CabanaPF;
TEST(PFHub1a, Initialization) {
    PfHubProblem simulation(96);    //96x96 grid
    simulation.fill_initial();
    //check 4 points for basic indexing/results:
    EXPECT_DOUBLE_EQ(0.53, simulation.get_c(0, 0));
    EXPECT_DOUBLE_EQ(0.48803644628427617, simulation.get_c(40, 0));
    EXPECT_DOUBLE_EQ(0.4926015127137723, simulation.get_c(0, 40));
    EXPECT_DOUBLE_EQ(0.5104736440583328, simulation.get_c(40, 40));
    //check 11 randomly selected points:
    EXPECT_DOUBLE_EQ(0.4939295245623995, simulation.get_c(18, 58));
    EXPECT_DOUBLE_EQ(0.5052812881659137, simulation.get_c(58, 22));
    EXPECT_DOUBLE_EQ(0.507573456034358, simulation.get_c(90, 50));
    EXPECT_DOUBLE_EQ(0.4937247597664062, simulation.get_c(93, 44));
    EXPECT_DOUBLE_EQ(0.49632841079525414, simulation.get_c(55, 64));
    EXPECT_DOUBLE_EQ(0.515161280104923, simulation.get_c(14, 68));
    EXPECT_DOUBLE_EQ(0.508133362029189, simulation.get_c(15, 10));
    EXPECT_DOUBLE_EQ(0.5031768252080895, simulation.get_c(94, 58));
    EXPECT_DOUBLE_EQ(0.5013653249684705, simulation.get_c(33, 6));
    EXPECT_DOUBLE_EQ(0.5173210860478162, simulation.get_c(84, 82));
    EXPECT_DOUBLE_EQ(0.48901386854420836, simulation.get_c(26, 42));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );
        return RUN_ALL_TESTS();
    }
    MPI_Finalize();
}