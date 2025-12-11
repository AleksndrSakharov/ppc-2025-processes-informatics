#include <gtest/gtest.h>

#include <cstddef>  // для size_t
#include <tuple>
#include <vector>

#include "pylaeva_s_max_elem_matrix/common/include/common.hpp"
#include "pylaeva_s_max_elem_matrix/mpi/include/ops_mpi.hpp"
#include "pylaeva_s_max_elem_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pylaeva_s_max_elem_matrix {

class PylaevaSMaxElemMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{0, 0, {}};
  const size_t kMatrixRows_ = 10000;
  const size_t kMatrixColumns_ = 10000;
  const size_t kTotalElements_ = kMatrixRows_ * kMatrixColumns_;
  OutType expected_data_ = static_cast<OutType>(static_cast<int>(kTotalElements_) - 1);

  void SetUp() override {
    std::vector<int> matrix_data;
    matrix_data.reserve(kTotalElements_);

    for (size_t i = 0; i < kTotalElements_; i++) {
      matrix_data.push_back(static_cast<int>(i));
    }

    input_data_ = std::make_tuple(kMatrixRows_, kMatrixColumns_, matrix_data);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data == expected_data_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PylaevaSMaxElemMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PylaevaSMaxElemMatrixMPI, PylaevaSMaxElemMatrixSEQ>(
    PPC_SETTINGS_pylaeva_s_max_elem_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PylaevaSMaxElemMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PylaevaSMaxElemMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace pylaeva_s_max_elem_matrix
