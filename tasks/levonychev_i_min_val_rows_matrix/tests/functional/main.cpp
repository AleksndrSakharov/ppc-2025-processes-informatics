#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"
#include "levonychev_i_min_val_rows_matrix/mpi/include/ops_mpi.hpp"
#include "levonychev_i_min_val_rows_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_min_val_rows_matrix {

class LevonychevIMinValRowsMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    bool reverse = std::get<0>(test_param);
    std::string flag;
    if (reverse) {
      flag = "reverse";
    } else {
      flag = "noreverse";
    }
    return flag + "_" + std::to_string(std::get<1>(test_param)) + "_" + std::to_string(std::get<2>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    bool reverse = std::get<0>(params);
    int rows = std::get<1>(params);
    int cols = std::get<2>(params);
    std::vector<int> matrix(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    output_data_.resize(static_cast<size_t>(rows));
    if (reverse) {
      for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (rows * cols) - (i + 1);
      }
      for (int i = 0; i < rows; ++i) {
        output_data_[i] = (rows * cols) - ((i + 1) * cols);
      }
    } else {
      for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = i;
      }
      for (int i = 0; i < rows; ++i) {
        output_data_[i] = i * cols;
      }
    }
    input_data_ = std::make_tuple(matrix, rows, cols);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == output_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType output_data_;
};

namespace {

TEST_P(LevonychevIMinValRowsMatrixFuncTests, MinValRowsMatrix) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {std::make_tuple(false, 1, 1),  std::make_tuple(false, 2, 1),
                                             std::make_tuple(false, 4, 3),  std::make_tuple(false, 3, 4),
                                             std::make_tuple(true, 6, 2),   std::make_tuple(false, 2, 6),
                                             std::make_tuple(true, 12, 1),  std::make_tuple(true, 1, 12),
                                             std::make_tuple(true, 10, 10), std::make_tuple(false, 10, 10)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LevonychevIMinValRowsMatrixMPI, InType>(
                                               kTestParam, PPC_SETTINGS_levonychev_i_min_val_rows_matrix),
                                           ppc::util::AddFuncTask<LevonychevIMinValRowsMatrixSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_levonychev_i_min_val_rows_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    LevonychevIMinValRowsMatrixFuncTests::PrintFuncTestName<LevonychevIMinValRowsMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(MinValRowsMatrixTests, LevonychevIMinValRowsMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace levonychev_i_min_val_rows_matrix
