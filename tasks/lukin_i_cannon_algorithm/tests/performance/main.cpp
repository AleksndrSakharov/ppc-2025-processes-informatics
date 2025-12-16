#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "lukin_i_cannon_algorithm/common/include/common.hpp"
#include "lukin_i_cannon_algorithm/mpi/include/ops_mpi.hpp"
#include "lukin_i_cannon_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lukin_i_cannon_algorithm {
const double kEpsilon = 1e-9;

class LukinIRunPerfTestsProcesses2 : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_data_;

  const int matrix_size_ = 1000;

  void SetUp() override {
    std::vector<double> a(static_cast<size_t>(matrix_size_ * matrix_size_));
    std::vector<double> b(static_cast<size_t>(matrix_size_ * matrix_size_));

    const int size = static_cast<int>(a.size());
    for (int i = 0; i < size; i++) {
      a[i] = i;
      b[i] = ((size - 1) - i);
    }

    std::vector<double> a_copy = a;
    std::vector<double> b_copy = b;

    input_data_ = std::make_tuple(std::move(a_copy), std::move(b_copy), matrix_size_);

    std::vector<double> c(static_cast<size_t>(matrix_size_ * matrix_size_));

    for (int i = 0; i < matrix_size_; i++) {
      for (int k = 0; k < matrix_size_; k++) {
        double fixed = a[(i * matrix_size_) + k];
        for (int j = 0; j < matrix_size_; j++) {
          c[(i * matrix_size_) + j] += fixed * b[(k * matrix_size_) + j];
        }
      }
    }
    expected_data_ = std::move(c);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const int size = static_cast<int>(expected_data_.size());
    for (int i = 0; i < size; i++) {
      if (std::abs(expected_data_[i] - output_data[i]) > kEpsilon) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LukinIRunPerfTestsProcesses2, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LukinICannonAlgorithmMPI, LukinICannonAlgorithmSEQ>(
    PPC_SETTINGS_lukin_i_cannon_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LukinIRunPerfTestsProcesses2::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunTests, LukinIRunPerfTestsProcesses2, kGtestValues, kPerfTestName);

}  // namespace lukin_i_cannon_algorithm
