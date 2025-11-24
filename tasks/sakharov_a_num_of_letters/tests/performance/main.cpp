#include <gtest/gtest.h>

#include "sakharov_a_num_of_letters/common/include/common.hpp"
#include "sakharov_a_num_of_letters/mpi/include/ops_mpi.hpp"
#include "sakharov_a_num_of_letters/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sakharov_a_num_of_letters {

class SakharovARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kRepeat = 10000000;
  const std::string pattern_ = "abcdABCD";
  InType input_data_{};
  OutType expected_result_ = 0;

  void SetUp() override {
    std::string s;
    s.reserve(static_cast<std::size_t>(kRepeat) * pattern_.size());
    for (int i = 0; i < kRepeat; i++) {
      s += pattern_;
    }
    int length = static_cast<int>(s.size());
    input_data_ = InType{length, s};
    expected_result_ = length;
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
    ppc::util::MakeAllPerfTasks<InType, SakharovANumberOfLettersMPI, SakharovANumberOfLettersSEQ>(
        PPC_SETTINGS_sakharov_a_num_of_letters);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SakharovARunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SakharovARunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace sakharov_a_num_of_letters
