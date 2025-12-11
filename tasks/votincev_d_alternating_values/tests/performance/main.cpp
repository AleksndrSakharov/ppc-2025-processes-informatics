#include <gtest/gtest.h>

#include <vector>

#include "util/include/perf_test_util.hpp"
#include "votincev_d_alternating_values/common/include/common.hpp"
#include "votincev_d_alternating_values/mpi/include/ops_mpi.hpp"
#include "votincev_d_alternating_values/seq/include/ops_seq.hpp"

namespace votincev_d_alternating_values {

class VotincevDAlternatigValuesRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  const int kCount_ = 10000000 + 1;
  OutType expected_res_ = 10000000;

  void SetUp() override {
    int vect_size = kCount_;
    std::vector<double> vect_data;
    int swapper = 1;
    for (int i = 0; i < vect_size; i++) {
      vect_data.push_back(i * swapper);  // 0 -1 2 -3 4 -5...
      swapper *= -1;
    }

    input_data_ = vect_data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // процессы (кроме 0-го) не сверяются с ответом
    if (output_data == -1) {
      return true;
    }

    // 0й процесс должен иметь правильный результат
    return output_data == expected_res_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, VotincevDAlternatingValuesMPI, VotincevDAlternatingValuesSEQ>(
        PPC_SETTINGS_votincev_d_alternating_values);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VotincevDAlternatigValuesRunPerfTestsProcesses::CustomPerfTestName;

TEST_P(VotincevDAlternatigValuesRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RunPerf, VotincevDAlternatigValuesRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace votincev_d_alternating_values
