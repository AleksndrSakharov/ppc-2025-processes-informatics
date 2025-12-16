#include <gtest/gtest.h>

#include <vector>

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"
#include "sakharov_a_cannon_algorithm/mpi/include/ops_mpi.hpp"
#include "sakharov_a_cannon_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sakharov_a_cannon_algorithm {

class SakharovARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kCount = 10000000;
  InType input_data_;
  OutType expected_result_;

  void SetUp() override {
    std::vector<int> data(kCount, 1);
    input_data_ = InType{0, data};
    expected_result_ = data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_result_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SakharovARunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SakharovACannonAlgorithmMPI, SakharovACannonAlgorithmSEQ>(
        PPC_SETTINGS_sakharov_a_cannon_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SakharovARunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SakharovARunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace sakharov_a_cannon_algorithm
