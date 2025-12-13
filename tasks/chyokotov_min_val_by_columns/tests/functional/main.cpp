#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <climits>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "chyokotov_min_val_by_columns/common/include/common.hpp"
#include "chyokotov_min_val_by_columns/mpi/include/ops_mpi.hpp"
#include "chyokotov_min_val_by_columns/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chyokotov_min_val_by_columns {

class ChyokotovMinValFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &matrix = std::get<0>(test_param);
    if (matrix.empty()) {
      return "empty_matrix";
    }
    return "size_of_matrix_" + std::to_string(matrix.size()) + "x" + std::to_string(matrix[0].size());
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(ChyokotovMinValFuncTest, MinValuesByColumnsOfMatrix) {
  ExecuteTest(GetParam());
}

const std::vector<std::vector<int>> kEmptyMatrix = {};
const std::vector<int> kExpectedEmpty = {};

const std::vector<std::vector<int>> kOneElement = {{0}};
const std::vector<int> kExpectedOneElement = {0};

const std::vector<std::vector<int>> kSqurMatrix = {{0, 1}, {-1, 0}};
const std::vector<int> kExpectedSqurMatrix = {-1, 0};

const std::vector<std::vector<int>> kRecMatrix = {{0, 1, 2, 3}, {-1, 0, 1, 2}};
const std::vector<int> kExpectedrecMatrix = {-1, 0, 1, 2};

const std::vector<std::vector<int>> kMaxMatrix = {{INT_MAX, INT_MAX}, {INT_MAX, INT_MAX}, {INT_MAX, INT_MAX}};
const std::vector<int> kExpectedMaxMatrix = {INT_MAX, INT_MAX};

const std::vector<std::vector<int>> kMinMatrix = {{INT_MIN, INT_MIN, INT_MIN}, {INT_MIN, INT_MIN, INT_MIN}};
const std::vector<int> kExpectedMinMatrix = {INT_MIN, INT_MIN, INT_MIN};

const std::vector<std::vector<int>> kOneColumn = {{0}, {1}, {2}, {3}, {4}};
const std::vector<int> kExpectedOneColumn = {0};

const std::vector<std::vector<int>> kOneRow = {{0, 1, 2, 3, 4}};
const std::vector<int> kExpectedOneRow = {0, 1, 2, 3, 4};

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(kEmptyMatrix, kExpectedEmpty),     std::make_tuple(kOneElement, kExpectedOneElement),
    std::make_tuple(kSqurMatrix, kExpectedSqurMatrix), std::make_tuple(kRecMatrix, kExpectedrecMatrix),
    std::make_tuple(kMaxMatrix, kExpectedMaxMatrix),   std::make_tuple(kMinMatrix, kExpectedMinMatrix),
    std::make_tuple(kOneColumn, kExpectedOneColumn),   std::make_tuple(kOneRow, kExpectedOneRow)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ChyokotovMinValByColumnsMPI, InType>(kTestParam, PPC_SETTINGS_chyokotov_min_val_by_columns),
    ppc::util::AddFuncTask<ChyokotovMinValByColumnsSEQ, InType>(kTestParam, PPC_SETTINGS_chyokotov_min_val_by_columns));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ChyokotovMinValFuncTest::PrintFuncTestName<ChyokotovMinValFuncTest>;

INSTANTIATE_TEST_SUITE_P(MinValuesTests, ChyokotovMinValFuncTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace chyokotov_min_val_by_columns
