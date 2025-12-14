#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "nikitina_v_quick_sort_merge/common/include/common.hpp"
#include "nikitina_v_quick_sort_merge/mpi/include/ops_mpi.hpp"
#include "nikitina_v_quick_sort_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nikitina_v_quick_sort_merge {

using TestParams = std::vector<int>;

class RunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestParams> {
 public:
  static std::string PrintTestParam(const TestParams &p) {
    if (p.empty()) {
      return "Empty";
    }
    std::string val = std::to_string(p[0]);
    if (p[0] < 0) {
      val = "Neg" + std::to_string(std::abs(p[0]));
    }
    return "Size_" + std::to_string(p.size()) + "_First_" + val;
  }

 protected:
  void SetUp() override {
    input_data_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (rank == 0) {
      std::vector<int> sorted_ref = input_data_;
      if (!sorted_ref.empty()) {
        QuickSortImpl(sorted_ref, 0, static_cast<int>(sorted_ref.size()) - 1);
      }
      return output_data == sorted_ref;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

std::vector<int> GenerateRandomVector(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-1000, 1000);
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

TEST_P(RunFuncTests, SortCheck) {
  ExecuteTest(GetParam());
}

const std::array<TestParams, 11> kTestVectors = {TestParams{},
                                                 TestParams{1},
                                                 TestParams{5, 4, 3, 2, 1},
                                                 TestParams{1, 2, 3, 4, 5, 6},
                                                 TestParams{2, 2, 2, 2},
                                                 TestParams{-5, -1, -3, -2, -4},
                                                 TestParams{-10, 0, 10, -5, 5, -1, 1},
                                                 TestParams{INT_MAX, INT_MIN, 0},
                                                 GenerateRandomVector(10),
                                                 GenerateRandomVector(50),
                                                 GenerateRandomVector(100)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TestTaskMPI, InType>(kTestVectors, PPC_SETTINGS_nikitina_v_quick_sort_merge),
                   ppc::util::AddFuncTask<TestTaskSEQ, InType>(kTestVectors, PPC_SETTINGS_nikitina_v_quick_sort_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = RunFuncTests::PrintFuncTestName<RunFuncTests>;

// NOLINTBEGIN(cert-err58-cpp,cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits,misc-use-anonymous-namespace)
INSTANTIATE_TEST_SUITE_P(QuickSortTests, RunFuncTests, kGtestValues, kPerfTestName);
// NOLINTEND(cert-err58-cpp,cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits,misc-use-anonymous-namespace)

}  // namespace

}  // namespace nikitina_v_quick_sort_merge
