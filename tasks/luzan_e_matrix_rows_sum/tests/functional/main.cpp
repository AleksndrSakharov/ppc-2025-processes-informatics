#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "luzan_e_matrix_rows_sum/common/include/common.hpp"
#include "luzan_e_matrix_rows_sum/mpi/include/ops_mpi.hpp"
#include "luzan_e_matrix_rows_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace luzan_e_matrix_rows_sum {

class LuzanEMatrixRowsSumFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<int>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int height = std::get<0>(params);
    int width = std::get<1>(params);
    std::tuple_element_t<0, InType> mat(static_cast<size_t>(height) * static_cast<size_t>(width));

    for (int elem = 0; elem < height * width; elem++) {
      mat[elem] = (elem * 2) - 42;
    }

    input_data_ = std::make_tuple(mat, height, width);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int height = std::get<1>(input_data_);
    int width = std::get<2>(input_data_);
    std::vector<int> sum(height, 0);
    std::tuple_element_t<0, InType> mat = std::get<0>(input_data_);

    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        sum[row] += mat[(width * row) + col];
      }
    }

    return (output_data == sum);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(LuzanEMatrixRowsSumFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {
    std::make_tuple(3, 3),    std::make_tuple(2, 5),    std::make_tuple(10, 70),
    std::make_tuple(2000, 5), std::make_tuple(5, 2000), std::make_tuple(1, 1),
    std::make_tuple(1, 100),  std::make_tuple(100, 1),  std::make_tuple(1000, 1000)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<LuzanEMatrixRowsSumMPI, InType>(kTestParam, PPC_SETTINGS_luzan_e_matrix_rows_sum),
    ppc::util::AddFuncTask<LuzanEMatrixRowsSumSEQ, InType>(kTestParam, PPC_SETTINGS_luzan_e_matrix_rows_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LuzanEMatrixRowsSumFuncTests::PrintFuncTestName<LuzanEMatrixRowsSumFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, LuzanEMatrixRowsSumFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luzan_e_matrix_rows_sum
