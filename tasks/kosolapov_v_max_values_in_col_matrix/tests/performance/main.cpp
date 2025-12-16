#include <gtest/gtest.h>

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"
#include "kosolapov_v_max_values_in_col_matrix/mpi/include/ops_mpi.hpp"
#include "kosolapov_v_max_values_in_col_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

class KosolapovVMaxValuesInColMatrixRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrix_size_ = 2000;
  InType input_data_;
  OutType expected_res_;

  void SetUp() override {
    input_data_.resize(kMatrix_size_);
    expected_res_.resize(kMatrix_size_);
    for (int i = 0; i < kMatrix_size_; i++) {
      input_data_[i].resize(kMatrix_size_);
      for (int j = 0; j < kMatrix_size_; j++) {
        input_data_[i][j] = i;
      }
      expected_res_[i] = kMatrix_size_ - 1;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_res_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KosolapovVMaxValuesInColMatrixRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KosolapovVMaxValuesInColMatrixMPI, KosolapovVMaxValuesInColMatrixSEQ>(
        PPC_SETTINGS_kosolapov_v_max_values_in_col_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KosolapovVMaxValuesInColMatrixRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerfTests, KosolapovVMaxValuesInColMatrixRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kosolapov_v_max_values_in_col_matrix
