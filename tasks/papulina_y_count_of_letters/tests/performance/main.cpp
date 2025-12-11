#include <gtest/gtest.h>

#include <string>

#include "papulina_y_count_of_letters/common/include/common.hpp"
#include "papulina_y_count_of_letters/mpi/include/ops_mpi.hpp"
#include "papulina_y_count_of_letters/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace papulina_y_count_of_letters {

class PapulinaYRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000000;
  const std::string s_ = "abcdabcd";
  InType input_data_;
  OutType expectedResult_ = 0;

  void SetUp() override {
    for (int i = 0; i < kCount_; i++) {
      input_data_ += s_;
    }
    expectedResult_ = static_cast<OutType>(kCount_ * s_.size());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expectedResult_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PapulinaYRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PapulinaYCountOfLettersMPI, PapulinaYCountOfLettersSEQ>(
    PPC_SETTINGS_papulina_y_count_of_letters);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PapulinaYRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PapulinaYRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace papulina_y_count_of_letters
