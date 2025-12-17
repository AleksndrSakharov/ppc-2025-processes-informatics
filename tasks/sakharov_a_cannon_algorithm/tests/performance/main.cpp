#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <vector>

#include "sakharov_a_cannon_algorithm/common/include/common.hpp"
#include "sakharov_a_cannon_algorithm/mpi/include/ops_mpi.hpp"
#include "sakharov_a_cannon_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

// Prevent gtest from printing std::function internals in PerfTestParam and causing valgrind noise.
namespace ppc::util {
template <typename InType, typename OutType>
static inline void PrintTo(const PerfTestParam<InType, OutType> &param, ::std::ostream *os) {
  *os << "PerfTestParam{"
      << "name=" << std::get<static_cast<std::size_t>(GTestParamIndex::kNameTest)>(param) << "}";
}
}  // namespace ppc::util

namespace sakharov_a_cannon_algorithm {

class SakharovARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_result_;

  void SetUp() override {
    const int size = SelectSize();
    input_data_ = BuildInput(size);
    expected_result_ = NaiveMultiply(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (expected_result_.size() != output_data.size()) {
      return false;
    }

    constexpr double kEps = 1e-9;
    for (std::size_t i = 0; i < expected_result_.size(); ++i) {
      if (std::abs(expected_result_[i] - output_data[i]) > kEps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  static int SelectSize() {
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const int grid_dim = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(world_size))));
    constexpr int kBlock = 128;
    return grid_dim * kBlock;
  }

  static InType BuildInput(int size) {
    InType in{};
    in.size = size;
    in.a.resize(static_cast<std::size_t>(size) * static_cast<std::size_t>(size));
    in.b.resize(static_cast<std::size_t>(size) * static_cast<std::size_t>(size));

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        in.a[Offset(size, i, j)] = static_cast<double>((i + 1) * (j + 2));
        in.b[Offset(size, i, j)] = (i == j) ? 1.0 : static_cast<double>((j % 3) + 1);
      }
    }

    return in;
  }

  static OutType NaiveMultiply(const InType &input) {
    const int n = input.size;
    OutType result(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);

    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < n; ++k) {
        const double a_val = input.a[Offset(n, i, k)];
        for (int j = 0; j < n; ++j) {
          result[Offset(n, i, j)] += a_val * input.b[Offset(n, k, j)];
        }
      }
    }

    return result;
  }
};

TEST_P(SakharovARunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SakharovACannonAlgorithmMPI, SakharovACannonAlgorithmSEQ>(
        PPC_SETTINGS_sakharov_a_cannon_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SakharovARunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SakharovARunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace sakharov_a_cannon_algorithm
