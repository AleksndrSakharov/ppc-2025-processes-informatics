#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "dorofeev_i_monte_carlo_integration/common/include/common.hpp"
#include "dorofeev_i_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "dorofeev_i_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dorofeev_i_monte_carlo_integration_processes {
namespace {

class MonteCarloPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  MonteCarloPerfTests() = default;

 private:
  // samples for performance test
  static constexpr int kSamples = 190000;

  InType input_data_{};

  void SetUp() override {
    input_data_.a = {0.0};
    input_data_.b = {1.0};
    input_data_.samples = kSamples;
    input_data_.func = [](const std::vector<double> &x) {
      return x[0] * x[0];  // integrating x^2 on [0,1]
    };
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double expected = 1.0 / 3.0;
    return std::abs(output_data - expected) < 0.05;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MonteCarloPerfTests, PerfTestModes) {
  ExecuteTest(GetParam());
}

// making performance probs: MPI + SEQ
const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DorofeevIMonteCarloIntegrationMPI, DorofeevIMonteCarloIntegrationSEQ>(
        PPC_SETTINGS_example_processes);

// converting to GTest values
const auto kGTestValues = ppc::util::TupleToGTestValues(kPerfTasks);

// test naming function
const auto kTestName = MonteCarloPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(MonteCarloPerf, MonteCarloPerfTests, kGTestValues, kTestName);

}  // namespace
}  // namespace dorofeev_i_monte_carlo_integration_processes
