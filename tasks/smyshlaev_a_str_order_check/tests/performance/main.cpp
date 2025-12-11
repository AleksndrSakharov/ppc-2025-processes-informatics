#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <utility>

#include "smyshlaev_a_str_order_check/common/include/common.hpp"
#include "smyshlaev_a_str_order_check/mpi/include/ops_mpi.hpp"
#include "smyshlaev_a_str_order_check/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace smyshlaev_a_str_order_check {

const size_t kStringLength = 20000000;

class SmyshlaevAStrOrderCheckRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  const OutType kExpectedResult_ = -1;
  InType input_data_;

  void SetUp() override {
    std::string long_str_a(kStringLength, 'a');
    std::string long_str_b = long_str_a;
    long_str_b.back() = 'b';
    input_data_ = std::make_pair(long_str_a, long_str_b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return kExpectedResult_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SmyshlaevAStrOrderCheckRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SmyshlaevAStrOrderCheckMPI, SmyshlaevAStrOrderCheckSEQ>(
    PPC_SETTINGS_smyshlaev_a_str_order_check);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SmyshlaevAStrOrderCheckRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SmyshlaevAStrOrderCheckRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace smyshlaev_a_str_order_check
