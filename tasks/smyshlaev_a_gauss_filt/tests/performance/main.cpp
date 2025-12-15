#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "smyshlaev_a_gauss_filt/common/include/common.hpp"
#include "smyshlaev_a_gauss_filt/mpi/include/ops_mpi.hpp"
#include "smyshlaev_a_gauss_filt/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace smyshlaev_a_gauss_filt {

class SmyshlaevAGaussFiltRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    int size = 1024;
    input_data_.width = size;
    input_data_.height = size;
    input_data_.channels = 3;
    input_data_.data.resize(static_cast<size_t>(size) * size * 3);

    for (size_t i = 0; i < input_data_.data.size(); ++i) {
      input_data_.data[i] = static_cast<uint8_t>(i % 256);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_.width == output_data.width && input_data_.height == output_data.height &&
            input_data_.channels == output_data.channels && !output_data.data.empty());
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SmyshlaevAGaussFiltRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SmyshlaevAGaussFiltMPI, SmyshlaevAGaussFiltSEQ>(
    PPC_SETTINGS_smyshlaev_a_gauss_filt);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SmyshlaevAGaussFiltRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SmyshlaevAGaussFiltRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace smyshlaev_a_gauss_filt
