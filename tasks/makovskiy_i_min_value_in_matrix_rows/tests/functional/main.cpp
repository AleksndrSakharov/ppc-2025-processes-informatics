#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "makovskiy_i_min_value_in_matrix_rows/common/include/common.hpp"
#include "makovskiy_i_min_value_in_matrix_rows/mpi/include/ops_mpi.hpp"
#include "makovskiy_i_min_value_in_matrix_rows/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

class MinValueRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &input = std::get<0>(test_param);
    const std::size_t rows = input.size();
    const std::size_t cols = input.empty() ? 0 : input.front().size();
    return "r" + std::to_string(rows) + "x" + std::to_string(cols);
  }

 protected:
  InType GetTestInputData() final {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    return std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }

    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const auto &expected = std::get<1>(params);
    return output_data == expected;
  }
};

TEST_P(MinValueRunFuncTests, MinPerRow) {
  ExecuteTest(GetParam());
}

namespace {

const auto kTestCases = std::array<TestType, 6>{
    TestType{InType{{1, 2, 3}, {4, 5, 6}}, OutType{1, 4}},
    TestType{InType{{-1, 0}, {10, 2}, {7}}, OutType{-1, 2, 7}},
    TestType{InType{{5, 5, 5}}, OutType{5}},
    TestType{InType{{8}}, OutType{8}},

    TestType{InType{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}}, OutType{1, 6, 11}},
    TestType{InType{{100, 200, 300}, {50, 60, 70}, {10, 20, 30}, {5, 6, 7}}, OutType{100, 50, 10, 5}},
};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<MinValueSEQ, InType>(kTestCases, PPC_SETTINGS_makovskiy_i_min_value_in_matrix_rows),
    ppc::util::AddFuncTask<MinValueMPI, InType>(kTestCases, PPC_SETTINGS_makovskiy_i_min_value_in_matrix_rows));

const auto kGtestValues = ppc::util::ExpandToValues(kTasks);
const auto kPerfTestName = MinValueRunFuncTests::PrintFuncTestName<MinValueRunFuncTests>;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(MinValueTests, MinValueRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace makovskiy_i_min_value_in_matrix_rows
