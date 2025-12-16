#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "liulin_y_matrix_max_column/common/include/common.hpp"
#include "liulin_y_matrix_max_column/mpi/include/ops_mpi.hpp"
#include "liulin_y_matrix_max_column/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace liulin_y_matrix_max_column {

class LiulinYMatrixMaxColumnFuncTestsFromFile : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &p) {
    return std::get<1>(p);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string filename = std::get<1>(params);

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_liulin_y_matrix_max_column, filename);

    std::ifstream file(abs_path + ".txt");
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open test file: " + abs_path + ".txt");
    }

    int rows = 0;
    int cols = 0;
    file >> rows >> cols;

    input_data_ = InType(rows, std::vector<int>(cols));

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        file >> input_data_[i][j];
      }
    }

    exp_output_ = OutType(cols, std::numeric_limits<int>::min());
    for (int ct = 0; ct < cols; ct++) {
      for (int rt = 0; rt < rows; rt++) {
        exp_output_[ct] = std::max(exp_output_[ct], input_data_[rt][ct]);
      }
    }

    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == exp_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType exp_output_;
};

namespace {

TEST_P(LiulinYMatrixMaxColumnFuncTestsFromFile, MaxByColumnsFromFile) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kTestParam = {
    std::make_tuple(0, "tinyMatrix"),      std::make_tuple(1, "simpleMatrix"), std::make_tuple(2, "randomMatrix"),
    std::make_tuple(3, "bigMatrix"),       std::make_tuple(4, "emptyMatrix"),  std::make_tuple(5, "singleElement"),
    std::make_tuple(6, "singleRow"),       std::make_tuple(7, "singleColumn"), std::make_tuple(8, "zeroColumns"),
    std::make_tuple(9, "negativeNumbers"), std::make_tuple(10, "allSame"),     std::make_tuple(11, "invalidMatrix")};
const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<LiulinYMatrixMaxColumnMPI, InType>(kTestParam, PPC_SETTINGS_liulin_y_matrix_max_column),
    ppc::util::AddFuncTask<LiulinYMatrixMaxColumnSEQ, InType>(kTestParam, PPC_SETTINGS_liulin_y_matrix_max_column));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName =
    LiulinYMatrixMaxColumnFuncTestsFromFile::PrintFuncTestName<LiulinYMatrixMaxColumnFuncTestsFromFile>;

INSTANTIATE_TEST_SUITE_P(FileTests, LiulinYMatrixMaxColumnFuncTestsFromFile, kGtestValues, kFuncTestName);

TEST(TournamentTestMPI, EmptyColumn) {
  std::vector<int> col;
  int res = LiulinYMatrixMaxColumnMPI::TournamentMax(col);
  ASSERT_EQ(res, std::numeric_limits<int>::min());
}

}  // namespace
}  // namespace liulin_y_matrix_max_column
