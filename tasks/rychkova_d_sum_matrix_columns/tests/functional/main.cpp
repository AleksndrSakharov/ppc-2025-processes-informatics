#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "rychkova_d_sum_matrix_columns/common/include/common.hpp"
#include "rychkova_d_sum_matrix_columns/mpi/include/ops_mpi.hpp"
#include "rychkova_d_sum_matrix_columns/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace rychkova_d_sum_matrix_columns {

class RychkovaRunFuncTestsMatrixColumns : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_matrix_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
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
    return input_matrix_;
  }

 private:
  InType input_matrix_;
  OutType expected_output_;
};

namespace {

TEST_P(RychkovaRunFuncTestsMatrixColumns, SumMatrixColumnsTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kTestParam = {
    std::make_tuple(std::vector<std::vector<int>>{{5}}, std::vector<int>{5}, "1x1_matrix"),
    std::make_tuple(std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}}, std::vector<int>{5, 7, 9}, "2x3_matrix"),
    std::make_tuple(std::vector<std::vector<int>>{{1, -2}, {-3, 4}, {5, -6}}, std::vector<int>{3, -4},
                    "3x2_matrix_with_negatives"),

    std::make_tuple(std::vector<std::vector<int>>{{0, 0, 0}, {0, 0, 0}}, std::vector<int>{0, 0, 0}, "zeros_2x3"),

    std::make_tuple(std::vector<std::vector<int>>{{1}, {2}, {3}, {4}}, std::vector<int>{10}, "4x1_matrix"),

    std::make_tuple(std::vector<std::vector<int>>{{1, 2}}, std::vector<int>{1, 2}, "1x2_matrix"),

    std::make_tuple(std::vector<std::vector<int>>{{1000000000, 1}, {1, 1000000000}},
                    std::vector<int>{1000000001, 1000000001}, "large_values"),

    std::make_tuple(std::vector<std::vector<int>>{{-1000000000, -1}, {-1, -1000000000}},
                    std::vector<int>{-1000000001, -1000000001}, "large_negative_values"),

    std::make_tuple(std::vector<std::vector<int>>{{100, 200, 300}, {-100, -200, -300}, {50, -50, 0}},
                    std::vector<int>{50, -50, 0}, "mixed_values_3x3"),

    std::make_tuple(std::vector<std::vector<int>>{{1, 2, 3, 4, 5}}, std::vector<int>{1, 2, 3, 4, 5}, "wide_matrix_1x5"),

    std::make_tuple(std::vector<std::vector<int>>{{1}, {2}, {3}, {4}, {5}}, std::vector<int>{15}, "tall_matrix_5x1"),

    std::make_tuple(std::vector<std::vector<int>>{{-1, -2, -3}, {-4, -5, -6}}, std::vector<int>{-5, -7, -9},
                    "all_negative_2x3")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<RychkovaDSumMatrixColumnsMPI, InType>(
                                               kTestParam, PPC_SETTINGS_rychkova_d_sum_matrix_columns),
                                           ppc::util::AddFuncTask<RychkovaDSumMatrixColumnsSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_rychkova_d_sum_matrix_columns));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RychkovaRunFuncTestsMatrixColumns::PrintFuncTestName<RychkovaRunFuncTestsMatrixColumns>;

INSTANTIATE_TEST_SUITE_P(MatrixColumnsTests, RychkovaRunFuncTestsMatrixColumns, kGtestValues, kPerfTestName);

class RychkovaInvalidMatrixTest : public ::testing::TestWithParam<InType> {
 protected:
  static void TestInvalidMatrix() {
    const InType &input_matrix = GetParam();

    RychkovaDSumMatrixColumnsMPI mpi_task(input_matrix);
    RychkovaDSumMatrixColumnsSEQ seq_task(input_matrix);

    EXPECT_FALSE(mpi_task.Validation());
    EXPECT_FALSE(seq_task.Validation());

    EXPECT_TRUE(mpi_task.GetOutput().empty());
    EXPECT_TRUE(seq_task.GetOutput().empty());
  }
};

TEST_P(RychkovaInvalidMatrixTest, ShouldFailValidationForInvalidMatrices) {
  RychkovaInvalidMatrixTest::TestInvalidMatrix();
}

const std::array<InType, 3> kInvalidMatrices = {std::vector<std::vector<int>>{{1, 2}, {3, 4, 5}, {6}},
                                                std::vector<std::vector<int>>{{1}, {2, 3}},
                                                std::vector<std::vector<int>>{{1, 2, 3}, {4, 5}}};

INSTANTIATE_TEST_SUITE_P(InvalidMatrixTests, RychkovaInvalidMatrixTest, ::testing::ValuesIn(kInvalidMatrices));

}  // namespace

}  // namespace rychkova_d_sum_matrix_columns
