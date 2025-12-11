#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "makovskiy_i_min_value_in_matrix_rows/common/include/common.hpp"
#include "makovskiy_i_min_value_in_matrix_rows/mpi/include/ops_mpi.hpp"
#include "makovskiy_i_min_value_in_matrix_rows/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

class MinValuePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType GetTestInputData() final {
    // constexpr std::size_t kRows = 100;
    // constexpr std::size_t kCols = 100;
    // constexpr std::size_t kRows = 1000;
    // constexpr std::size_t kCols = 1000;
    // constexpr std::size_t kRows = 5000;
    // constexpr std::size_t kCols = 5000;
    constexpr std::size_t kRows = 10000;
    constexpr std::size_t kCols = 10000;
    InType mat(kRows, std::vector<int>(kCols));
    for (std::size_t i = 0; i < kRows; ++i) {
      for (std::size_t j = 0; j < kCols; ++j) {
        mat[i][j] = static_cast<int>(i + j);
      }
    }
    return mat;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }

    // return output_data.size() == 100;
    // return output_data.size() == 1000;
    // return output_data.size() == 5000;
    return output_data.size() == 10000;
  }
};

TEST_P(MinValuePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MinValueSEQ, MinValueMPI>(PPC_SETTINGS_makovskiy_i_min_value_in_matrix_rows);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MinValuePerfTests::CustomPerfTestName;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(MinValuePerf, MinValuePerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace makovskiy_i_min_value_in_matrix_rows
