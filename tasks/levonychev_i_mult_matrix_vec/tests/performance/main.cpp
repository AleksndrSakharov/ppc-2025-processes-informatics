#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "levonychev_i_mult_matrix_vec/common/include/common.hpp"
#include "levonychev_i_mult_matrix_vec/mpi/include/ops_mpi.hpp"
#include "levonychev_i_mult_matrix_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace levonychev_i_mult_matrix_vec {

class LevonychevIMultMatrixVecPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  int ROWS_ = 4096;
  int COLS_ = 4096;
  InType input_data_;
  OutType expected_result_;
  void SetUp() override {
    std::vector<double> matrix(static_cast<size_t>(ROWS_) * static_cast<size_t>(COLS_));
    std::vector<double> x(COLS_);
    for (int j = 0; j < COLS_; ++j) {
      x[j] = static_cast<double>((j) % 100) * 0.57932;
    }
    for (int i = 0; i < ROWS_; ++i) {
      for (int j = 0; j < COLS_; ++j) {
        matrix[(i * COLS_) + j] = static_cast<double>((i + j) % 100) * 89.56916;
      }
    }
    input_data_ = std::make_tuple(matrix, ROWS_, COLS_, x);
    expected_result_.resize(ROWS_);
    for (int i = 0; i < ROWS_; ++i) {
      double scalar_product = 0;
      for (int j = 0; j < COLS_; ++j) {
        scalar_product += matrix[(i * COLS_) + j] * x[j];
      }
      expected_result_[i] = scalar_product;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_result_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LevonychevIMultMatrixVecPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LevonychevIMultMatrixVecMPI, LevonychevIMultMatrixVecSEQ>(
        PPC_SETTINGS_levonychev_i_mult_matrix_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LevonychevIMultMatrixVecPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LevonychevIMultMatrixVecPerfTests, kGtestValues, kPerfTestName);

}  // namespace levonychev_i_mult_matrix_vec
