#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "rozenberg_a_matrix_column_sum/common/include/common.hpp"
#include "rozenberg_a_matrix_column_sum/mpi/include/ops_mpi.hpp"
#include "rozenberg_a_matrix_column_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace rozenberg_a_matrix_column_sum {

class RozenbergAMatrixColumnSumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    input_data_.clear();
    output_data_.clear();
    if (CheckTestAndRank()) {
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_rozenberg_a_matrix_column_sum, "perf_test.txt");
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

TEST_P(RozenbergAMatrixColumnSumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RozenbergAMatrixColumnSumMPI, RozenbergAMatrixColumnSumSEQ>(
        PPC_SETTINGS_rozenberg_a_matrix_column_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RozenbergAMatrixColumnSumPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RozenbergAMatrixColumnSumPerfTests, kGtestValues, kPerfTestName);

}  // namespace rozenberg_a_matrix_column_sum
