#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "kurpiakov_a_elem_vec_sum/common/include/common.hpp"
#include "kurpiakov_a_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "kurpiakov_a_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kurpiakov_a_elem_vec_sum {

class KurpiakovAElemVecSumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{0, {}};
  OutType expected_data_{0};

  void SetUp() override {
    const int vector_size = 10'000'000;
    std::vector<int> input(vector_size);

    for (int i = 0; i < vector_size; i++) {
      input[i] = i + 1;
    }

    expected_data_ = static_cast<OutType>(vector_size) * (vector_size + 1) / 2;
    input_data_ = InType(vector_size, input);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data == expected_data_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

namespace {

TEST_P(KurpiakovAElemVecSumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KurpiakovAElemVecSumMPI, KurpiakovAElemVecSumSEQ>(
    PPC_SETTINGS_kurpiakov_a_elem_vec_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KurpiakovAElemVecSumPerfTests::CustomPerfTestName;

// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(KurpiakovAVecPerf, KurpiakovAElemVecSumPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kurpiakov_a_elem_vec_sum
