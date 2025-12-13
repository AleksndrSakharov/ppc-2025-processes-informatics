#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <tuple>

#include "romanov_a_integration_rect_method/common/include/common.hpp"
#include "romanov_a_integration_rect_method/mpi/include/ops_mpi.hpp"
#include "romanov_a_integration_rect_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace romanov_a_integration_rect_method {

class RomanovAIntegrationRectMethodPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100'000'000;
  InType input_data_;

  void SetUp() override {
    auto f = std::function<double(double)>([](double x) { return std::pow(x, 3); });
    double a = -10.0;
    double b = 10.0;
    int n = kCount_;

    input_data_ = std::make_tuple(f, a, b, n);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return IsEqual(static_cast<double>(output_data), 0.0);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RomanovAIntegrationRectMethodPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RomanovAIntegrationRectMethodMPI, RomanovAIntegrationRectMethodSEQ>(
        PPC_SETTINGS_romanov_a_integration_rect_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RomanovAIntegrationRectMethodPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RomanovAIntegrationRectMethodPerfTests, kGtestValues, kPerfTestName);

}  // namespace romanov_a_integration_rect_method
