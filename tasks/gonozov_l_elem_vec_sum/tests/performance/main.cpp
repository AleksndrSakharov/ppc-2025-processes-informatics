#include <gtest/gtest.h>

#include <vector>

#include "gonozov_l_elem_vec_sum/common/include/common.hpp"
#include "gonozov_l_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "gonozov_l_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gonozov_l_elem_vec_sum {

class GonozovLElemVecSumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  int kCount_ = 100000000;
  InType input_data_;
  OutType desired_result_ = 0LL;

  void SetUp() override {
    input_data_ = std::vector<int>(kCount_);
    for (int i = 0; i < kCount_; i++) {
      input_data_[i] = i;
    }
    desired_result_ = static_cast<OutType>((kCount_ - 1)) * static_cast<OutType>(kCount_ / 2);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == desired_result_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GonozovLElemVecSumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, GonozovLElemVecSumMPI, GonozovLElemVecSumSEQ>(
    PPC_SETTINGS_gonozov_l_elem_vec_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GonozovLElemVecSumPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GonozovLElemVecSumPerfTests, kGtestValues, kPerfTestName);

}  // namespace gonozov_l_elem_vec_sum
