#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "shvetsova_k_max_diff_neig_vec/common/include/common.hpp"
#include "shvetsova_k_max_diff_neig_vec/mpi/include/ops_mpi.hpp"
#include "shvetsova_k_max_diff_neig_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shvetsova_k_max_diff_neig_vec {

class ShvetsovaKMaxDiffNeigVecRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const std::vector<double> kCount_ = {0.0};
  InType input_data_;
  OutType expect_res_;

  void SetUp() override {
    for (int i = 0; i < 10000000; i++) {
      if (i != 1) {
        input_data_.push_back(i);
      }
      if (i == 1) {
        input_data_.push_back(10000);
      }
    }
    expect_res_.first = input_data_[0];
    expect_res_.second = input_data_[1];
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double eps = 0.05;
    return (std::abs(output_data.first - expect_res_.first) <= eps &&
            std::abs(output_data.second - expect_res_.second) <= eps);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShvetsovaKMaxDiffNeigVecMPI, ShvetsovaKMaxDiffNeigVecSEQ>(
        PPC_SETTINGS_shvetsova_k_max_diff_neig_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShvetsovaKMaxDiffNeigVecRunPerfTestProcesses::CustomPerfTestName;

TEST_P(ShvetsovaKMaxDiffNeigVecRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShvetsovaKMaxDiffNeigVecRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace shvetsova_k_max_diff_neig_vec
