#include <gtest/gtest.h>

#include <cstddef>

#include "khruev_a_min_elem_vec/common/include/common.hpp"
#include "khruev_a_min_elem_vec/mpi/include/ops_mpi.hpp"
#include "khruev_a_min_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace khruev_a_min_elem_vec {

class KhruevAMinElemVecPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100000000;
  InType input_data_;
  // OutType expected_;

  void SetUp() override {
    size_t size = kCount_;
    for (size_t i = size; i >= 1; i--) {
      input_data_.push_back(static_cast<int>(i));
    }
    // expected_ = 1;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return 1 == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KhruevAMinElemVecPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KhruevAMinElemVecMPI, KhruevAMinElemVecSEQ>(PPC_SETTINGS_khruev_a_min_elem_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KhruevAMinElemVecPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerf, KhruevAMinElemVecPerfTests, kGtestValues, kPerfTestName);

}  // namespace khruev_a_min_elem_vec
