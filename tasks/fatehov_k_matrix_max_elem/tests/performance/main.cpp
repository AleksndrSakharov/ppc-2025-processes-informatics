#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "fatehov_k_matrix_max_elem/common/include/common.hpp"
#include "fatehov_k_matrix_max_elem/mpi/include/ops_mpi.hpp"
#include "fatehov_k_matrix_max_elem/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace fatehov_k_matrix_max_elem {

class FatehovKRunPerfTestsMatrixMaxElem : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_ = std::make_tuple(0, 0, std::vector<double>{});
  OutType expected_result_ = 0;

  void SetUp() override {
    const size_t rows = 5000;
    const size_t cols = 10000;
    const size_t total = rows * cols;

    uint64_t state = 42;
    const uint64_t a = 1664525ULL;
    const uint64_t c = 1013904223ULL;
    const uint64_t m = (1ULL << 22);

    std::vector<double> matrix;
    matrix.reserve(total);
    double max_val = -std::numeric_limits<double>::max();

    for (size_t i = 0; i < total; ++i) {
      state = (a * state + c) % m;
      double value = ((static_cast<double>(state) / m) * 2000.0) - 1000.0;
      matrix.push_back(value);

      max_val = std::max(value, max_val);
    }

    input_data_ = std::make_tuple(rows, cols, matrix);
    expected_result_ = max_val;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_result_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(FatehovKRunPerfTestsMatrixMaxElem, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, FatehovKMatrixMaxElemMPI, FatehovKMatrixMaxElemSEQ>(
    PPC_SETTINGS_fatehov_k_matrix_max_elem);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = FatehovKRunPerfTestsMatrixMaxElem::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerfTest, FatehovKRunPerfTestsMatrixMaxElem, kGtestValues, kPerfTestName);

}  // namespace fatehov_k_matrix_max_elem
