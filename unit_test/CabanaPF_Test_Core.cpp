#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <CommandLine.hpp>
#include <PFVariables.hpp>

using namespace CabanaPF;

TEST(PFVariables, saveload) {
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(std::array<double, 2>{0, 0}, std::array<double, 2>{6, 6},
                                                             std::array<int, 2>{3, 3});
    Cabana::Grid::DimBlockPartitioner<2> partitioner;
    auto global_grid =
        Cabana::Grid::createGlobalGrid(MPI_COMM_WORLD, global_mesh, std::array<bool, 2>{true, true}, partitioner);
    auto local_grid = Cabana::Grid::createLocalGrid(global_grid, 0);
    auto layout = createArrayLayout(local_grid, 2, Cabana::Grid::Node());
    PFVariables vars(layout, std::array<std::string, 1>{"a"});
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            vars[0](i, j, 0) = 10 * i + j;
            vars[0](i, j, 1) = -.0005;
        }
    }
    vars.save(0, "Test", 0);

    PFVariables from_file(layout, std::array<std::string, 1>{"a"});
    from_file.load(0, "Test", 0);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_EQ(vars[0](i, j, 0), from_file[0](i, j, 0));
            EXPECT_EQ(vars[0](i, j, 1), from_file[0](i, j, 1));
        }
    }
}

// helpers for testing command line arguments
char** create_argv(int argc, ...) {
    char** argv = new char*[argc + 1];
    va_list args;
    va_start(args, argc);
    for (int i = 0; i < argc; i++) {
        const char* arg = va_arg(args, const char*);
        argv[i] = new char[strlen(arg) + 1];
        strcpy(argv[i], arg);
    }
    argv[argc] = nullptr;
    va_end(args);
    return argv;
}
void destroy_argv(char** argv) {
    for (int i = 0; argv[i] != nullptr; i++)
        delete[] argv[i];
    delete[] argv;
}

TEST(CommandLineInput, Parsing) {
    char** argv =
        create_argv(17, "./Test", "--grid", "40", "--endoutput", "15", "--dt", ".14159", "--log", "--majoroutputs", "6",
                    "--minoroutputs", "60", "--startoutput", "10", "--endtime", "225.289", "--outputatzero");
    CommandLineInput input;
    input.read_command_line(17, argv);
    ASSERT_EQ(40, input.grid_points);
    ASSERT_NEAR(.14159, input.dt, 1e-9);
    ASSERT_NEAR(225.289, input.end_time, 1e-9);
    ASSERT_EQ(6, input.major_outputs);
    ASSERT_EQ(60, input.minor_outputs);
    ASSERT_EQ(10, input.start_output);
    ASSERT_EQ(15, input.end_output);
    ASSERT_TRUE(input.log_scale);
    ASSERT_TRUE(input.output_at_zero);
    destroy_argv(argv);
}

TEST(CommandLineInput, Verifying) {
    char** argv;
    // Missing required argument:
    argv = create_argv(5, "./Test", "--grid", "250", "--dt", ".1");
    ASSERT_THROW(
        {
            CommandLineInput input;
            input.read_command_line(5, argv);
        },
        std::invalid_argument);
    destroy_argv(argv);

    // Bad output range:
    argv = create_argv(11, "./Test", "--grid", "250", "--dt", ".1", "--endtime", "250", "--startoutput", "300",
                       "--minoroutputs", "100");
    ASSERT_THROW(
        {
            CommandLineInput input;
            input.read_command_line(11, argv);
        },
        std::invalid_argument);
    destroy_argv(argv);

    // Log-scale starting at 0
    argv =
        create_argv(5, "./Test", "--grid", "250", "--dt", ".1", "--endtime", "250", "--log", "--minoroutputs", "100");
    ASSERT_THROW(
        {
            CommandLineInput input;
            input.read_command_line(5, argv);
        },
        std::invalid_argument);
    destroy_argv(argv);
}

class OutputTester : public CabanaPFRunner<2> {
  public:
    int init_called = 0;
    int n_major_outputs = 0;
    int n_minor_outputs = 0;

    void initialize() override {
        init_called = true;
    }
    void step() override {}
    void minor_output() override {
        n_minor_outputs++;
    }
    void major_output() override {
        n_major_outputs++;
    }

    OutputTester() : CabanaPFRunner(10, 10, .1) {}
};

TEST(Output, Output) {
    OutputTester output_tester;
    output_tester.add_output(3, true);
    output_tester.add_output(3, false);
    output_tester.add_output(5, true);
    output_tester.add_output(3, false);
    output_tester.add_output(0, false);
    // should have initialized automatically due to outputting at t=0:
    ASSERT_EQ(1, output_tester.init_called);
    ASSERT_EQ(1, output_tester.n_minor_outputs);
    ASSERT_EQ(0, output_tester.n_major_outputs);
    // do a full run:
    output_tester.run_for_time(5);
    ASSERT_EQ(1, output_tester.init_called);
    ASSERT_EQ(3, output_tester.n_minor_outputs);
    ASSERT_EQ(2, output_tester.n_major_outputs);
}

TEST(CommandLineInput, set_runner_outputs) {
    char** argv = create_argv(15, "./Test", "--grid", "40", "--startoutput", "2", "--endoutput", "10", "--dt", ".1",
                              "--majoroutputs", "2", "--minoroutputs", "5", "--endtime", "10");
    CommandLineInput input;
    input.read_command_line(15, argv);

    OutputTester output_tester;
    input.add_outputs_to_runner(output_tester);
    // partway through:
    output_tester.run_for_time(4);
    ASSERT_EQ(1, output_tester.n_major_outputs);
    ASSERT_EQ(2, output_tester.n_minor_outputs);
    // end:
    output_tester.run_until_time(10);
    ASSERT_EQ(2, output_tester.n_major_outputs);
    ASSERT_EQ(5, output_tester.n_minor_outputs);
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
