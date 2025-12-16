#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kutuzov_i_elem_vec_average/common/include/common.hpp"
#include "kutuzov_i_elem_vec_average/mpi/include/ops_mpi.hpp"
#include "kutuzov_i_elem_vec_average/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kutuzov_i_elem_vec_average {

class KutuzovIElemVecAveragePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 50000000;
  InType input_data_;
  double answer_ = 0.0;

  void SetUp() override {
    input_data_ = std::vector<double>(kCount_, 0.0);

    answer_ = 0.0;
    for (int i = 0; i < kCount_; i++) {
      double value = static_cast<double>(i * i) - (static_cast<double>(kCount_) / 2.0);
      input_data_[i] = value;
      answer_ += value;
    }
    answer_ /= static_cast<double>(kCount_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool result = abs(output_data - answer_) < abs(answer_ * 0.0001);

    return result;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KutuzovIElemVecAveragePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KutuzovIElemVecAverageMPI, KutuzovIElemVecAverageSEQ>(
    PPC_SETTINGS_kutuzov_i_elem_vec_average);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KutuzovIElemVecAveragePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KutuzovIElemVecAveragePerfTests, kGtestValues, kPerfTestName);

}  // namespace kutuzov_i_elem_vec_average
