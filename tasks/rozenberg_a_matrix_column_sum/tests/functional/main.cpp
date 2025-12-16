#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "rozenberg_a_matrix_column_sum/common/include/common.hpp"
#include "rozenberg_a_matrix_column_sum/mpi/include/ops_mpi.hpp"
#include "rozenberg_a_matrix_column_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace rozenberg_a_matrix_column_sum {

class RozenbergAMatrixColumnFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    input_data_.clear();
    output_data_.clear();
    if (CheckTestAndRank()) {
      TestType filename =
          std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam()) + ".txt";
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_rozenberg_a_matrix_column_sum, filename);
      std::ifstream file(abs_path);

      if (file.is_open()) {
        int rows = 0;
        int columns = 0;
        file >> rows >> columns;

        InType input_data(rows, std::vector<int>(columns));
        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < columns; j++) {
            file >> input_data[i][j];
          }
        }

        OutType output_data(columns);
        for (int i = 0; i < columns; i++) {
          file >> output_data[i];
        }
        input_data_ = input_data;
        output_data_ = output_data;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (CheckTestAndRank()) {
      return (output_data_ == output_data);
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType output_data_;

  static bool CheckTestAndRank() {
    const std::string &test_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());
    return (test_name.find("mpi") == std::string::npos || !ppc::util::IsUnderMpirun() || ppc::util::GetMPIRank() == 0);
  }
};

namespace {

TEST_P(RozenbergAMatrixColumnFuncTests, MatrixColumnSum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {"basic_test",         "large_test",        "same_value_test",
                                            "single_column_test", "single_row_test",   "single_value_test",
                                            "random_data_test",   "random_data_test_2"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<RozenbergAMatrixColumnSumMPI, InType>(
                                               kTestParam, PPC_SETTINGS_rozenberg_a_matrix_column_sum),
                                           ppc::util::AddFuncTask<RozenbergAMatrixColumnSumSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_rozenberg_a_matrix_column_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RozenbergAMatrixColumnFuncTests::PrintFuncTestName<RozenbergAMatrixColumnFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, RozenbergAMatrixColumnFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace rozenberg_a_matrix_column_sum
