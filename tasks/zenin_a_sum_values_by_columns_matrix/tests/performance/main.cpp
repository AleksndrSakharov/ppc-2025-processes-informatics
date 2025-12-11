#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zenin_a_sum_values_by_columns_matrix/common/include/common.hpp"
#include "zenin_a_sum_values_by_columns_matrix/mpi/include/ops_mpi.hpp"
#include "zenin_a_sum_values_by_columns_matrix/seq/include/ops_seq.hpp"

namespace zenin_a_sum_values_by_columns_matrix {

class ZeninASumValuesByMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t rows_ = 6000;
  const size_t cols_ = 6000;
  InType input_data_;

  void SetUp() override {
    std::vector<double> mat(rows_ * cols_);
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < cols_; j++) {
        mat[(i * cols_) + j] = static_cast<double>((i + j) % 1000);
      }
    }
    input_data_ = std::make_tuple(rows_, cols_, std::move(mat));
  }
  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<double> expected(cols_, 0.0);
    const auto &mat = std::get<2>(input_data_);
    for (size_t j = 0; j < cols_; j++) {
      for (size_t i = 0; i < rows_; i++) {
        expected[j] += mat[(i * cols_) + j];
      }
    }
    return (output_data == expected);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZeninASumValuesByMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ZeninASumValuesByColumnsMatrixMPI, ZeninASumValuesByColumnsMatrixSEQ>(
        PPC_SETTINGS_zenin_a_sum_values_by_columns_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZeninASumValuesByMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ZeninAPerfTestMatrix, ZeninASumValuesByMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace zenin_a_sum_values_by_columns_matrix
