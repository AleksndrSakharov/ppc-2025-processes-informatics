#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"
#include "kosolapov_v_max_values_in_col_matrix/mpi/include/ops_mpi.hpp"
#include "kosolapov_v_max_values_in_col_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

class KosolapovVMaxValuesInColMatrixRunFuncTestsProcesses
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string data_source =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_kosolapov_v_max_values_in_col_matrix, params + ".txt");
    std::ifstream file(data_source);
    int rows = 0;
    file >> rows;
    int columns = 0;
    file >> columns;
    std::vector<std::vector<int>> input_matrix(rows, std::vector<int>(columns));
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        int elem = 0;
        file >> elem;
        input_matrix[i][j] = elem;
      }
    }
    std::vector<int> vector_max(columns);
    for (int i = 0; i < columns; i++) {
      int elem = 0;
      file >> elem;
      vector_max[i] = elem;
    }
    input_data_ = input_matrix;
    expected_res_ = vector_max;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_res_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_res_;
};

namespace {

TEST_P(KosolapovVMaxValuesInColMatrixRunFuncTestsProcesses, MaxInMatrixColumns) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 13> kTestParam = {"big_matrix",
                                             "matrix_1_to_1",
                                             "matrix_with_1_column",
                                             "matrix_with_1_row",
                                             "matrix_with_big_elements",
                                             "matrix_with_big_negative_elems",
                                             "matrix_with_negative_element",
                                             "matrix_with_same_element",
                                             "matrix_with_unequal_number_rows_and_columns",
                                             "max_in_first_rows",
                                             "max_in_last_rows",
                                             "max_on_diagonal",
                                             "simple_matrix"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KosolapovVMaxValuesInColMatrixMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kosolapov_v_max_values_in_col_matrix),
                                           ppc::util::AddFuncTask<KosolapovVMaxValuesInColMatrixSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kosolapov_v_max_values_in_col_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KosolapovVMaxValuesInColMatrixRunFuncTestsProcesses::PrintFuncTestName<
    KosolapovVMaxValuesInColMatrixRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MaxInMatrixColumns, KosolapovVMaxValuesInColMatrixRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace kosolapov_v_max_values_in_col_matrix
