#include <gtest/gtest.h>
#include <mpi.h>

#include <random>
#include <vector>

#include "nikitina_v_quick_sort_merge/common/include/common.hpp"
#include "nikitina_v_quick_sort_merge/mpi/include/ops_mpi.hpp"
#include "nikitina_v_quick_sort_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitina_v_quick_sort_merge {

class RunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int count = 1000000;
    input_data_.resize(count);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-100000, 100000);
    for (int &val : input_data_) {
      val = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return true;
    }

    std::vector<int> ref = input_data_;
    if (!ref.empty()) {
      QuickSortImpl(ref, 0, static_cast<int>(ref.size()) - 1);
    }
    return output_data == ref;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(RunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TestTaskMPI, TestTaskSEQ>(PPC_SETTINGS_nikitina_v_quick_sort_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = RunPerfTests::CustomPerfTestName;

// NOLINTBEGIN(cert-err58-cpp,cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits,misc-use-anonymous-namespace)
INSTANTIATE_TEST_SUITE_P(QuickSortPerfTests, RunPerfTests, kGtestValues, kPerfTestName);
// NOLINTEND(cert-err58-cpp,cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits,misc-use-anonymous-namespace)

}  // namespace nikitina_v_quick_sort_merge
