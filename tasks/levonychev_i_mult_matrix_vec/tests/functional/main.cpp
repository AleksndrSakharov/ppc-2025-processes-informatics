#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "levonychev_i_mult_matrix_vec/common/include/common.hpp"
#include "levonychev_i_mult_matrix_vec/mpi/include/ops_mpi.hpp"
#include "levonychev_i_mult_matrix_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_mult_matrix_vec {

class LevonychevIMultMatrixVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int rows = std::get<0>(params);
    const int cols = std::get<1>(params);
    std::vector<double> matrix(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    std::vector<double> x(cols);
    for (int i = 0; i < rows * cols; ++i) {
      matrix[i] = static_cast<double>(i + 1);
    }
    for (int i = 0; i < cols; ++i) {
      x[i] = static_cast<double>(i + 1);
    }
    input_data_ = std::make_tuple(matrix, rows, cols, x);
    output_data_.resize(rows);
    for (int i = 0; i < rows; ++i) {
      double scalar_product = 0.0;
      for (int j = 0; j < cols; ++j) {
        scalar_product += matrix[(i * cols) + j] * x[j];
      }
      output_data_[i] = scalar_product;
    }
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

TEST_P(LevonychevIMultMatrixVecFuncTests, MultMatrixVec) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {std::make_tuple(6, 6), std::make_tuple(4, 2), std::make_tuple(2, 4),
                                            std::make_tuple(1, 1), std::make_tuple(2, 1), std::make_tuple(1, 2),
                                            std::make_tuple(2, 2), std::make_tuple(3, 7)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<LevonychevIMultMatrixVecMPI, InType>(kTestParam, PPC_SETTINGS_levonychev_i_mult_matrix_vec),
    ppc::util::AddFuncTask<LevonychevIMultMatrixVecSEQ, InType>(kTestParam, PPC_SETTINGS_levonychev_i_mult_matrix_vec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LevonychevIMultMatrixVecFuncTests::PrintFuncTestName<LevonychevIMultMatrixVecFuncTests>;

INSTANTIATE_TEST_SUITE_P(MultMatrixVecTests, LevonychevIMultMatrixVecFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace levonychev_i_mult_matrix_vec
