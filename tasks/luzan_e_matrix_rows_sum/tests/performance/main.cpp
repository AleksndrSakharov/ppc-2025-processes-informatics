#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "luzan_e_matrix_rows_sum/common/include/common.hpp"
#include "luzan_e_matrix_rows_sum/mpi/include/ops_mpi.hpp"
#include "luzan_e_matrix_rows_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luzan_e_matrix_rows_sum {

class LuzanEMatrixRowsSumpERFTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int height_ = 12500;
  const int width_ = 12500;
  InType input_data_;

  void SetUp() override {
    std::tuple_element_t<0, InType> mat(static_cast<size_t>(height_) * static_cast<size_t>(width_));
    for (int elem = 0; elem < height_ * width_; elem++) {
      mat[elem] = elem % 42000;
    }

    input_data_ = std::make_tuple(mat, height_, width_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<int> sum(height_);
    std::tuple_element_t<0, InType> mat = std::get<0>(input_data_);

    for (int row = 0; row < height_; row++) {
      for (int col = 0; col < width_; col++) {
        sum[row] += mat[(width_ * row) + col];
      }
    }

    return (output_data == sum);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LuzanEMatrixRowsSumpERFTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LuzanEMatrixRowsSumMPI, LuzanEMatrixRowsSumSEQ>(
    PPC_SETTINGS_luzan_e_matrix_rows_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuzanEMatrixRowsSumpERFTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuzanEMatrixRowsSumpERFTests, kGtestValues, kPerfTestName);

}  // namespace luzan_e_matrix_rows_sum
