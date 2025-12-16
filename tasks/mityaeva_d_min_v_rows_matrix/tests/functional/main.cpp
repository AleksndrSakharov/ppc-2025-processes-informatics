#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>

#include "mityaeva_d_min_v_rows_matrix/common/include/common.hpp"
#include "mityaeva_d_min_v_rows_matrix/mpi/include/ops_mpi.hpp"
#include "mityaeva_d_min_v_rows_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace mityaeva_d_min_v_rows_matrix {

class MinValuesInRowsRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_index = std::get<0>(params);
    switch (test_index) {
      case 1: {
        input_data_ = {3, 3, 1, 2, 3, 4, 0, 6, 7, 8, 9};
        expected_output_ = {3, 1, 0, 7};
        break;
      }
      case 2: {
        input_data_ = {1, 4, 5, 3, 8, 1};
        expected_output_ = {1, 1};
        break;
      }
      case 3: {
        input_data_ = {3, 1, 2, 5, 1};
        expected_output_ = {3, 2, 5, 1};
        break;
      }
      case 4: {
        input_data_ = {2, 3, -1, -5, 3, 2, 0, -2};
        expected_output_ = {2, -5, -2};
        break;
      }
      case 5: {
        input_data_ = {1, 1, 10};
        expected_output_ = {1, 10};
        break;
      }
      case 6: {
        input_data_ = {4, 4, 15, 12, 18, 11, 9, 14, 7, 16, 13, 8, 19, 10, 17, 6, 20, 5};
        expected_output_ = {4, 11, 7, 8, 5};
        break;
      }
      default:
        throw std::runtime_error("Unknown test index: " + std::to_string(test_index));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(MinValuesInRowsRunFuncTests, MinValuesInRows) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(1, "simple_3x3_matrix"),    std::make_tuple(2, "single_row_matrix"),
    std::make_tuple(3, "single_column_matrix"), std::make_tuple(4, "negative_numbers"),
    std::make_tuple(5, "single_element"),       std::make_tuple(6, "large_4x4_matrix")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MinValuesInRowsMPI, InType>(kTestParam, PPC_SETTINGS_mityaeva_d_min_v_rows_matrix),
    ppc::util::AddFuncTask<MinValuesInRowsSEQ, InType>(kTestParam, PPC_SETTINGS_mityaeva_d_min_v_rows_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = MinValuesInRowsRunFuncTests::PrintFuncTestName<MinValuesInRowsRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixMinValuesTests, MinValuesInRowsRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace mityaeva_d_min_v_rows_matrix
