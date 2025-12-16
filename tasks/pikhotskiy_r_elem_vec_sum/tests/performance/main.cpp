#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "pikhotskiy_r_elem_vec_sum/common/include/common.hpp"
#include "pikhotskiy_r_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "pikhotskiy_r_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pikhotskiy_r_elem_vec_sum {

class PerformanceBenchmark : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType benchmark_input_{0, {}};
  OutType expected_value_{0};

  void SetUp() override {
    const int element_count = 100'000'000;
    std::vector<int> test_data(element_count);

    for (int i = 0; i < element_count; i++) {
      if (i % 3 == 0) {
        test_data[i] = i + 1;
      } else if (i % 3 == 1) {
        test_data[i] = 42;
      } else {
        test_data[i] = element_count - i;
      }
    }

    expected_value_ = 0LL;
    for (int i = 0; i < element_count; i++) {
      if (i % 3 == 0) {
        expected_value_ += static_cast<OutType>(i + 1);
      } else if (i % 3 == 1) {
        expected_value_ += 42LL;
      } else {
        expected_value_ += static_cast<OutType>(element_count - i);
      }
    }

    benchmark_input_ = InType(element_count, test_data);
  }

  // NOLINTNEXTLINE(readability-make-member-function-const)
  bool CheckTestOutputData(OutType &actual_output) final {
    return actual_output == expected_value_;
  }

  InType GetTestInputData() final {
    return benchmark_input_;
  }
};

namespace {

TEST_P(PerformanceBenchmark, ExecutePerformanceTests) {
  ExecuteTest(GetParam());
}

const auto kPerformanceTasks = ppc::util::MakeAllPerfTasks<InType, PikhotskiyRElemVecSumMPI, PikhotskiyRElemVecSumSEQ>(
    PPC_SETTINGS_pikhotskiy_r_elem_vec_sum);

const auto kTestValues = ppc::util::TupleToGTestValues(kPerformanceTasks);

const auto kBenchmarkNaming = PerformanceBenchmark::CustomPerfTestName;

// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(VectorSumBenchmarks, PerformanceBenchmark, kTestValues, kBenchmarkNaming);

}  // namespace

}  // namespace pikhotskiy_r_elem_vec_sum
