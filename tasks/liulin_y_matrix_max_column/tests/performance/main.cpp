#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "liulin_y_matrix_max_column/common/include/common.hpp"
#include "liulin_y_matrix_max_column/mpi/include/ops_mpi.hpp"
#include "liulin_y_matrix_max_column/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace liulin_y_matrix_max_column {

class LiulinYMatrixMaxColumnPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  struct TestConfig {
    int rows;
    int cols;
    std::string name;
  };

  static TestConfig GetLiulinSpecificConfig(const std::string &test_name) {
    constexpr int kLargeDim = 2048;
    if (test_name.find("liulin_y_matrix_max_column_mpi") != std::string::npos) {
      return {.rows = kLargeDim, .cols = kLargeDim, .name = "mpi_large"};
    }
    if (test_name.find("liulin_y_matrix_max_column_seq") != std::string::npos) {
      return {.rows = kLargeDim, .cols = kLargeDim, .name = "seq_large"};
    }
    return {.rows = 100, .cols = 100, .name = "small"};
  }

  static bool HasLiulinSpecificConfig(const std::string &test_name) {
    return test_name.find("liulin_y_matrix_max_column") != std::string::npos;
  }

  static TestConfig GetTestConfig(const std::string &test_name) {
    if (HasLiulinSpecificConfig(test_name)) {
      return GetLiulinSpecificConfig(test_name);
    }
    if (test_name.find("small") != std::string::npos) {
      return {.rows = 100, .cols = 100, .name = "small"};
    }
    if (test_name.find("medium") != std::string::npos) {
      return {.rows = 1000, .cols = 1000, .name = "medium"};
    }
    if (test_name.find("large") != std::string::npos) {
      return {.rows = 10000, .cols = 10000, .name = "large"};
    }
    if (test_name.find("xlarge") != std::string::npos) {
      return {.rows = 20000, .cols = 20000, .name = "xlarge"};
    }
    if (test_name.find("tall") != std::string::npos) {
      return {.rows = 100, .cols = 5000, .name = "tall"};
    }
    if (test_name.find("wide") != std::string::npos) {
      return {.rows = 5000, .cols = 100, .name = "wide"};
    }
    return {.rows = 100, .cols = 100, .name = "small"};
  }

  void SetUp() override {
    const auto &test_param = GetParam();
    const std::string &test_name = std::get<1>(test_param);

    TestConfig config = GetTestConfig(test_name);
    rows_ = config.rows;
    cols_ = config.cols;

    GenerateTestData();
  }

  static int GenerateCellValue(int row, int col) {
    constexpr int kMod = 2001;
    const int raw_value = (row * 131 + col * 17 + kMod) % kMod;
    return raw_value - 1000;
  }

  void GenerateTestData() {
    input_data_.resize(rows_);
    for (auto &row : input_data_) {
      row.assign(cols_, 0);
    }

    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < cols_; j++) {
        input_data_[i][j] = GenerateCellValue(i, j);
      }
    }

    expected_output_.assign(cols_, std::numeric_limits<int>::min());
    for (int col = 0; col < cols_; col++) {
      for (int row = 0; row < rows_; row++) {
        expected_output_[col] = std::max(expected_output_[col], input_data_[row][col]);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (std::size_t i = 0; i < output_data.size(); ++i) {
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
  int rows_ = 0;
  int cols_ = 0;
  InType input_data_;
  OutType expected_output_;
};

TEST_P(LiulinYMatrixMaxColumnPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LiulinYMatrixMaxColumnMPI, LiulinYMatrixMaxColumnSEQ>(
    PPC_SETTINGS_liulin_y_matrix_max_column);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LiulinYMatrixMaxColumnPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LiulinYMatrixMaxColumnPerfTests, kGtestValues, kPerfTestName);

}  // namespace liulin_y_matrix_max_column
