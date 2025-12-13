#include <gtest/gtest.h>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"
#include "kiselev_i_max_value_in_strings/mpi/include/ops_mpi.hpp"
#include "kiselev_i_max_value_in_strings/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kiselev_i_max_value_in_strings {

class KiselevIMaxValueInStringsRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int kCount = 10000;    // количество строк
  static constexpr int kMaxCols = 10000;  // макс длина строки

  void SetUp() override {
    input_data_.resize(kCount);
    for (int i = 0; i < kCount; ++i) {
      int len_row = 1 + (i % kMaxCols);
      input_data_[i].resize(len_row);
      for (int j = 0; j < len_row; ++j) {
        input_data_[i][j] = (i + 2 + j + 7) % 100;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return !output_data.empty() && output_data.size() == input_data_.size();
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(KiselevIMaxValueInStringsRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KiselevITestTaskMPI, KiselevITestTaskSEQ>(
    PPC_SETTINGS_kiselev_i_max_value_in_strings);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KiselevIMaxValueInStringsRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KiselevIMaxValueInStringsRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kiselev_i_max_value_in_strings
