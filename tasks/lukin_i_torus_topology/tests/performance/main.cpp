#include <gtest/gtest.h>
#include <mpi.h>

#include <tuple>
#include <vector>

#include "lukin_i_torus_topology/common/include/common.hpp"
#include "lukin_i_torus_topology/mpi/include/ops_mpi.hpp"
#include "lukin_i_torus_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lukin_i_torus_topology {
const int kLargeVectorSize = 10'000'000;

class LukinIRunPerfTestProceses3 : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_result_;

  std::vector<int> dummy_ = {2, 0, 2, 6};
  std::vector<int> message_;

  void SetUp() override {
    message_ = std::vector<int>(kLargeVectorSize, 1);

    int proc_count = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    if (proc_count == 1) {
      input_data_ = std::make_tuple(0, 0, message_);
      expected_result_ = std::make_tuple(std::vector<int>{}, message_);
      return;
    }

    if (proc_count == 2) {
      input_data_ = std::make_tuple(0, 1, message_);
      expected_result_ = std::make_tuple(std::vector<int>{0, 1}, message_);
      return;
    }

    if (proc_count == 3) {
      input_data_ = std::make_tuple(0, 2, message_);
      expected_result_ = std::make_tuple(std::vector<int>{0, 2}, message_);
      return;
    }

    // proc_count == 4
    input_data_ = std::make_tuple(0, 3, message_);
    expected_result_ = std::make_tuple(std::vector<int>{0, 2, 3}, message_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<int> route = std::get<0>(output_data);
    if (route == dummy_) {  // seq
      return true;
    }

    return expected_result_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LukinIRunPerfTestProceses3, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LukinIThorTopologyMPI, LukinIThorTopologySEQ>(
    PPC_SETTINGS_lukin_i_torus_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LukinIRunPerfTestProceses3::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunTests, LukinIRunPerfTestProceses3, kGtestValues, kPerfTestName);

}  // namespace lukin_i_torus_topology
