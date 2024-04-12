#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <PFHub.hpp>
#include <PFVariables.hpp>

using namespace CabanaPF;

/*
PFHub1a benchmark problem (Spinodal Decomposition): https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/
These results that are being tested against come from the python implementation of a pseudospectral implementation of
PFHub1a Available here (ORNL internal): https://code.ornl.gov/71d/phase-field-example-codes Most points were randomly
selected
*/
TEST(PFHub1a, Initialization) {
    PFHub1aBenchmark simulation(96, 500);
    simulation.timestep(0); // trigger initialization
    auto results = simulation.get_cpu_view();
    //"true results" come from python implementation (see previous comment)
    // check 4 points for basic indexing/results:
    EXPECT_DOUBLE_EQ(0.53, results(0, 0, 0));
    EXPECT_DOUBLE_EQ(0.48803644628427617, results(40, 0, 0));
    EXPECT_DOUBLE_EQ(0.4926015127137723, results(0, 40, 0));
    EXPECT_DOUBLE_EQ(0.5104736440583328, results(40, 40, 0));
    // check 11 randomly selected points:
    EXPECT_DOUBLE_EQ(0.4939295245623995, results(18, 58, 0));
    EXPECT_DOUBLE_EQ(0.5052812881659137, results(58, 22, 0));
    EXPECT_DOUBLE_EQ(0.507573456034358, results(90, 50, 0));
    EXPECT_DOUBLE_EQ(0.4937247597664062, results(93, 44, 0));
    EXPECT_DOUBLE_EQ(0.49632841079525414, results(55, 64, 0));
    EXPECT_DOUBLE_EQ(0.515161280104923, results(14, 68, 0));
    EXPECT_DOUBLE_EQ(0.508133362029189, results(15, 10, 0));
    EXPECT_DOUBLE_EQ(0.5031768252080895, results(94, 58, 0));
    EXPECT_DOUBLE_EQ(0.5013653249684705, results(33, 6, 0));
    EXPECT_DOUBLE_EQ(0.5173210860478162, results(84, 82, 0));
    EXPECT_DOUBLE_EQ(0.48901386854420836, results(26, 42, 0));
}

TEST(PFHub1a, OneTimestep) {
    PFHub1aBenchmark simulation(96, 500);
    simulation.timestep(1);
    auto results = simulation.get_cpu_view();
    // test at extreme points and 10 random points.  Correct values come from python implementation (see above)
    EXPECT_DOUBLE_EQ(0.5214689225639189, results(0, 0, 0));
    EXPECT_DOUBLE_EQ(0.49527507173039187, results(95, 95, 0));
    EXPECT_DOUBLE_EQ(0.5076626764926879, results(15, 19, 0));
    EXPECT_DOUBLE_EQ(0.4827922294845011, results(85, 94, 0));
    EXPECT_DOUBLE_EQ(0.5067545231942185, results(44, 77, 0));
    EXPECT_DOUBLE_EQ(0.498421920827976, results(89, 78, 0));
    EXPECT_DOUBLE_EQ(0.5057763046363665, results(15, 71, 0));
    EXPECT_DOUBLE_EQ(0.5069675088240566, results(75, 38, 0));
    EXPECT_DOUBLE_EQ(0.5120896916735972, results(27, 55, 0));
    EXPECT_DOUBLE_EQ(0.5049713217512952, results(49, 81, 0));
    EXPECT_DOUBLE_EQ(0.5081090491604472, results(43, 76, 0));
    EXPECT_DOUBLE_EQ(0.5069592731273633, results(50, 11, 0));
}

TEST(PFHub1a, AllTimestep) {
    PFHub1aBenchmark simulation(96, 500);
    simulation.timestep(500);
    auto results = simulation.get_cpu_view();
    // as before, (0,0), (95,95), and 10 random points, testing against python
    EXPECT_NEAR(0.4412261765305555, results(0, 0, 0), 1e-9);
    EXPECT_NEAR(0.3470514937291973, results(95, 95, 0), 1e-9);
    EXPECT_NEAR(0.3042812431506664, results(31, 68, 0), 1e-9);
    EXPECT_NEAR(0.6942634682896683, results(16, 91, 0), 1e-9);
    EXPECT_NEAR(0.557699155088305, results(62, 21, 0), 1e-9);
    EXPECT_NEAR(0.3124243999273498, results(2, 79, 0), 1e-9);
    EXPECT_NEAR(0.694823598752751, results(73, 75, 0), 1e-9);
    EXPECT_NEAR(0.6770420862143505, results(40, 11, 0), 1e-9);
    EXPECT_NEAR(0.3423994469216821, results(85, 67, 0), 1e-9);
    EXPECT_NEAR(0.4279715141345520, results(40, 78, 0), 1e-9);
    EXPECT_NEAR(0.6966615561225524, results(70, 72, 0), 1e-9);
    EXPECT_NEAR(0.6482746495041702, results(67, 7, 0), 1e-9);
}

#ifdef RESULTS_PATH
TEST(PFVariables, saveload) {
    auto global_mesh = Cajita::createUniformGlobalMesh(std::array<double, 2>{0, 0}, std::array<double, 2>{6, 6},
                                                       std::array<int, 2>{3, 3});
    Cajita::DimBlockPartitioner<2> partitioner;
    auto global_grid =
        Cajita::createGlobalGrid(MPI_COMM_WORLD, global_mesh, std::array<bool, 2>{true, true}, partitioner);
    auto local_grid = Cajita::createLocalGrid(global_grid, 0);
    auto layout = createArrayLayout(local_grid, 2, Cajita::Node());
    PFVariables vars(layout, std::array<std::string, 1>{"a"});
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            vars[0](i, j, 0) = 10 * i + j;
            vars[0](i, j, 1) = -.0005;
        }
    }
    vars.save(0, "Test", 0);

    PFVariables from_file(layout, std::array<std::string, 1>{"a"});
    from_file.load("Test", 0);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_EQ(vars[0](i, j, 0), from_file[0](i, j, 0));
            EXPECT_EQ(vars[0](i, j, 1), from_file[0](i, j, 1));
        }
    }
}
#endif

// Similar to above, the python implementation was modified to use the periodic initial conditions
TEST(PFHub1aPeriodic, periodic) {
    PFHub1aPeriodic simulation(96, 500);
    simulation.timestep(0);
    auto results = simulation.get_cpu_view();
    EXPECT_NEAR(0.53, results(0, 0, 0), 1e-9);
    EXPECT_NEAR(0.5280872555770657, results(95, 95, 0), 1e-9);
    EXPECT_NEAR(0.49625, results(56, 52, 0), 1e-9);
    EXPECT_NEAR(0.5096103712433676, results(39, 36, 0), 1e-9);
    EXPECT_NEAR(0.5122826024701564, results(46, 40, 0), 1e-9);

    simulation.timestep(1);
    results = simulation.get_cpu_view();
    EXPECT_NEAR(0.5316722086053631, results(0, 0, 0), 1e-9);
    EXPECT_NEAR(0.5296339912527902, results(95, 95, 0), 1e-9);
    EXPECT_NEAR(0.5155424558547776, results(24, 46, 0), 1e-9);
    EXPECT_NEAR(0.510243048825588, results(87, 78, 0), 1e-9);
    EXPECT_NEAR(0.5092351158827323, results(6, 19, 0), 1e-9);

    simulation.timestep(499);
    results = simulation.get_cpu_view();
    EXPECT_NEAR(0.6993369106233298, results(0, 0, 0), 1e-9);
    EXPECT_NEAR(0.7014658707445363, results(95, 95, 0), 1e-9);
    EXPECT_NEAR(0.6427344294446387, results(0, 28, 0), 1e-9);
    EXPECT_NEAR(0.6076503841641254, results(35, 65, 0), 1e-9);
    EXPECT_NEAR(0.3520246964993546, results(74, 32, 0), 1e-9);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    int return_val = RUN_ALL_TESTS();

    Kokkos::finalize();
    MPI_Finalize();
    return return_val;
}
