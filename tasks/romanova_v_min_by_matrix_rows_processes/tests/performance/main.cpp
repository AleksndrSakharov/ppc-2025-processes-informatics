#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "romanova_v_min_by_matrix_rows_processes/common/include/common.hpp"
#include "romanova_v_min_by_matrix_rows_processes/mpi/include/ops_mpi.hpp"
#include "romanova_v_min_by_matrix_rows_processes/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace romanova_v_min_by_matrix_rows_processes {

class RomanovaVMinByMatrixRowsPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType exp_answer_;

  void SetUp() override {
    std::string path = "matrixForPerfTest.txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_romanova_v_min_by_matrix_rows_processes, path);
    std::ifstream file(abs_path);
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
};

TEST_P(RomanovaVMinByMatrixRowsPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RomanovaVMinByMatrixRowsMPI, RomanovaVMinByMatrixRowsSEQ>(
        PPC_SETTINGS_romanova_v_min_by_matrix_rows_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RomanovaVMinByMatrixRowsPerfTestProcesses::CustomPerfTestName;
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits, misc-use-anonymous-namespace)
INSTANTIATE_TEST_SUITE_P(RunModeTests, RomanovaVMinByMatrixRowsPerfTestProcesses, kGtestValues, kPerfTestName);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits, misc-use-anonymous-namespace)
}  // namespace romanova_v_min_by_matrix_rows_processes
