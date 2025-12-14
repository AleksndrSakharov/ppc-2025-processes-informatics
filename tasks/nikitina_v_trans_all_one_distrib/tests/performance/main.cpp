#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "nikitina_v_trans_all_one_distrib/common/include/common.hpp"
#include "nikitina_v_trans_all_one_distrib/mpi/include/ops_mpi.hpp"
#include "nikitina_v_trans_all_one_distrib/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitina_v_trans_all_one_distrib {

class NikitinaVRunPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    const int size = 20000000;
    input_data_ = std::vector<int>(size, 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }
    return output_data.size() == input_data_.size();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(NikitinaVRunPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {
const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TestTaskMPI, TestTaskSEQ>(PPC_SETTINGS_nikitina_v_trans_all_one_distrib);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NikitinaVRunPerfTest::CustomPerfTestName;

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits, misc-use-anonymous-namespace)
INSTANTIATE_TEST_SUITE_P(RunModeTests, NikitinaVRunPerfTest, kGtestValues, kPerfTestName);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits, misc-use-anonymous-namespace)

}  // namespace

}  // namespace nikitina_v_trans_all_one_distrib
