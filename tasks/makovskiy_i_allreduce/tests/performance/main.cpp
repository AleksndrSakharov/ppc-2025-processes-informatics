#include <gtest/gtest.h>

#include "makovskiy_i_allreduce/common/include/common.hpp"
#include "makovskiy_i_allreduce/mpi/include/ops_mpi.hpp"
#include "makovskiy_i_allreduce/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace makovskiy_i_allreduce {

class AllreducePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType GetTestInputData() final {
    constexpr int kCount = 100000000;
    InType data(static_cast<InType::size_type>(kCount), 1);
    return data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr int kCount = 100000000;
    return !output_data.empty() && output_data[0] == kCount;
  }
};

TEST_P(AllreducePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kPerfTestName = AllreducePerfTests::CustomPerfTestName;

const auto kSeqPerfTasks = ppc::util::MakeAllPerfTasks<InType, TestTaskSEQ>(PPC_SETTINGS_makovskiy_i_allreduce);
const auto kSeqGtestValues = ppc::util::TupleToGTestValues(kSeqPerfTasks);

const auto kMpiPerfTasks = ppc::util::MakeAllPerfTasks<InType, TestTaskMPI>(PPC_SETTINGS_makovskiy_i_allreduce);
const auto kMpiGtestValues = ppc::util::TupleToGTestValues(kMpiPerfTasks);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(AllreducePerfSEQ, AllreducePerfTests, kSeqGtestValues, kPerfTestName);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(AllreducePerfMPI, AllreducePerfTests, kMpiGtestValues, kPerfTestName);

}  // namespace

}  // namespace makovskiy_i_allreduce
