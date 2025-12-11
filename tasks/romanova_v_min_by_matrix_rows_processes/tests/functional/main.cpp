#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "romanova_v_min_by_matrix_rows_processes/common/include/common.hpp"
#include "romanova_v_min_by_matrix_rows_processes/mpi/include/ops_mpi.hpp"
#include "romanova_v_min_by_matrix_rows_processes/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace romanova_v_min_by_matrix_rows_processes {

class RomanovaVMinByMatrixRowsFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_romanova_v_min_by_matrix_rows_processes, params);
    std::ifstream file(abs_path + ".txt");
    if (file.is_open()) {
      int rows = 0;
      int columns = 0;
      file >> rows >> columns;
      exp_answer_ = OutType(rows);
      for (int i = 0; i < rows; i++) {
        file >> exp_answer_[i];
      }

      input_data_ = InType(rows, std::vector<int>(columns));
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
          file >> input_data_[i][j];
        }
      }

      file.close();
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (exp_answer_.size() != output_data.size()) {
      return false;
    }
    for (size_t i = 0; i < exp_answer_.size(); i++) {
      if (exp_answer_[i] != output_data[i]) {
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
  OutType exp_answer_;
};

namespace {

TEST_P(RomanovaVMinByMatrixRowsFuncTestsProcesses, MinByMatrixRowsFromFile) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    "reallySmallTest", "simpleTest", "averageTest", "increasingValuesInRowsTest", "decreasingValuesInRowsTest",
    "sameValuesTest"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<RomanovaVMinByMatrixRowsMPI, InType>(
                                               kTestParam, PPC_SETTINGS_romanova_v_min_by_matrix_rows_processes),
                                           ppc::util::AddFuncTask<RomanovaVMinByMatrixRowsSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_romanova_v_min_by_matrix_rows_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName =
    RomanovaVMinByMatrixRowsFuncTestsProcesses::PrintFuncTestName<RomanovaVMinByMatrixRowsFuncTestsProcesses>;
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(FileTests, RomanovaVMinByMatrixRowsFuncTestsProcesses, kGtestValues, kFuncTestName);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
}  // namespace

}  // namespace romanova_v_min_by_matrix_rows_processes
