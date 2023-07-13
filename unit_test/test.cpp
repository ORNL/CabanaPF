#include <gtest/gtest.h>
#include <Cabana_Core.hpp>
#include <Cajita.hpp>

#include <PFHub.hpp>
#include <PFVariables.hpp>
#include <simulation.hpp>

using namespace CabanaPF;

/*
PFHub1a benchmark problem (Spinodal Decomposition): https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/
These results that are being tested against come from the python implementation of a pseudospectral implentation of PFHub1a
    Available here (ORNL internal): https://code.ornl.gov/71d/phase-field-example-codes
    Most points were randomly selected
*/
TEST(PFHub1a, Initialization) {
    Simulation<PFHub1aBenchmark> simulation(96, 500);
    //"true results" come from python implentation (see previous comment)
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

TEST(PFHub1a, OneTimestep) {
    Simulation<PFHub1aBenchmark> simulation(96, 500);
    simulation.timestep(1);
    //test at extreme points and 10 random points.  Correct values come from python implemtation (see above)
    EXPECT_DOUBLE_EQ(0.5214689225639189, simulation.get_c(0, 0));
    EXPECT_DOUBLE_EQ(0.49527507173039187, simulation.get_c(95, 95));
    EXPECT_DOUBLE_EQ(0.5076626764926879, simulation.get_c(15, 19));
    EXPECT_DOUBLE_EQ(0.4827922294845011, simulation.get_c(85, 94));
    EXPECT_DOUBLE_EQ(0.5067545231942185, simulation.get_c(44, 77));
    EXPECT_DOUBLE_EQ(0.498421920827976, simulation.get_c(89, 78));
    EXPECT_DOUBLE_EQ(0.5057763046363665, simulation.get_c(15, 71));
    EXPECT_DOUBLE_EQ(0.5069675088240566, simulation.get_c(75, 38));
    EXPECT_DOUBLE_EQ(0.5120896916735972, simulation.get_c(27, 55));
    EXPECT_DOUBLE_EQ(0.5049713217512952, simulation.get_c(49, 81));
    EXPECT_DOUBLE_EQ(0.5081090491604472, simulation.get_c(43, 76));
    EXPECT_DOUBLE_EQ(0.5069592731273633, simulation.get_c(50, 11));
}

TEST(PFHub1a, AllTimestep) {
    Simulation<PFHub1aBenchmark> simulation(96, 500);
    simulation.timestep(500);
    //as before, (0,0), (95,95), and 10 random points, testing against python
    EXPECT_NEAR(0.4412261765305555, simulation.get_c(0, 0), 1e-9);
    EXPECT_NEAR(0.3470514937291973, simulation.get_c(95, 95), 1e-9);
    EXPECT_NEAR(0.3042812431506664, simulation.get_c(31, 68), 1e-9);
    EXPECT_NEAR(0.6942634682896683, simulation.get_c(16, 91), 1e-9);
    EXPECT_NEAR(0.557699155088305, simulation.get_c(62, 21), 1e-9);
    EXPECT_NEAR(0.31242439992734977, simulation.get_c(2, 79), 1e-9);
    EXPECT_NEAR(0.694823598752751, simulation.get_c(73, 75), 1e-9);
    EXPECT_NEAR(0.6770420862143505, simulation.get_c(40, 11), 1e-9);
    EXPECT_NEAR(0.3423994469216821, simulation.get_c(85, 67), 1e-9);
    EXPECT_NEAR(0.42797151413455203, simulation.get_c(40, 78), 1e-9);
    EXPECT_NEAR(0.6966615561225524, simulation.get_c(70, 72), 1e-9);
    EXPECT_NEAR(0.6482746495041702, simulation.get_c(67, 7), 1e-9);
}

TEST(PFVariables, saveload) {
    auto global_mesh = Cajita::createUniformGlobalMesh(
        std::array<double, 2> {0, 0},
        std::array<double, 2> {6, 6},
        std::array<int, 2> {3, 3}
    );
    Cajita::DimBlockPartitioner<2> partitioner;
    auto global_grid = Cajita::createGlobalGrid(MPI_COMM_WORLD, global_mesh, std::array<bool, 2>{true, true}, partitioner);
    auto local_grid = Cajita::createLocalGrid( global_grid, 0 );
    auto layout = createArrayLayout(local_grid, 2, Cajita::Node());
    PFVariables vars(layout, std::array<std::string, 1> {"a"});
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            vars[0](i, j, 0) = 10*i+j;
            vars[0](i, j, 1) = -.0005;
        }
    }
    vars.save(0, "Test", 0);

    PFVariables from_file(layout, std::array<std::string, 1> {"a"});
    from_file.load("Test", 0);
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            EXPECT_EQ(vars[0](i, j, 0), from_file[0](i, j, 0));
            EXPECT_EQ(vars[0](i, j, 1), from_file[0](i, j, 1));
        }
    }
}

//Similar to above, the python implmentation was modified to use the same periodic initial conditions
TEST(PFHub1aSimplePeriodic, periodic) {
    Simulation<PFHub1aSimplePeriodic> simul(96, 500);
    EXPECT_NEAR(0.52, simul.get_c(0, 0), 1e-8);
    EXPECT_NEAR(0.515, simul.get_c(40,0), 1e-8);
    EXPECT_NEAR(0.49633974596215563, simul.get_c(40,40), 1e-8);
    EXPECT_NEAR(0.5013397459621556, simul.get_c(0,40), 1e-8);

    simul.timestep(500);
    EXPECT_NEAR(0.6899556834423245, simul.get_c(0, 0), 1e-8);
    EXPECT_NEAR(0.710215729789399, simul.get_c(95, 95), 1e-8);
    EXPECT_NEAR(0.6203452063008421, simul.get_c(31, 68), 1e-8);
    EXPECT_NEAR(0.29988065061700286, simul.get_c(16, 91), 1e-8);
    EXPECT_NEAR(0.3425650624685481, simul.get_c(62, 21), 1e-8);
    EXPECT_NEAR(0.6352884582630192, simul.get_c(2, 79), 1e-8);
    EXPECT_NEAR(0.29039495705145013, simul.get_c(73, 75), 1e-8);
    EXPECT_NEAR(0.6102543428687541, simul.get_c(40, 11), 1e-8);
    EXPECT_NEAR(0.6245817469274588, simul.get_c(85, 67), 1e-8);
    EXPECT_NEAR(0.631637186261738, simul.get_c(40, 78), 1e-8);
    EXPECT_NEAR(0.35473143197267865, simul.get_c(70, 72), 1e-8);
    EXPECT_NEAR(0.6338012425033652, simul.get_c(67, 7), 1e-8);

    //test setup on bigger grid:
    Simulation<PFHub1aSimplePeriodic> big_grid(4*96, 500);
    EXPECT_NEAR(0.52, big_grid.get_c(0, 0), 1e-9);
    EXPECT_NEAR(0.515, big_grid.get_c(160,0), 1e-9);
    EXPECT_NEAR(0.49633974596215563, big_grid.get_c(160,160), 1e-9);
    EXPECT_NEAR(0.5013397459621556, big_grid.get_c(0,160), 1e-9);
}

int main(int argc, char** argv) {
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );
    ::testing::InitGoogleTest( &argc, argv );
    int return_val = RUN_ALL_TESTS();

    Kokkos::finalize();
    MPI_Finalize();
    return return_val;
}
