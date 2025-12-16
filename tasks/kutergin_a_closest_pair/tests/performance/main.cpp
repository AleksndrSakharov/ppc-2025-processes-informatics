#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "kutergin_a_closest_pair/common/include/common.hpp"
#include "kutergin_a_closest_pair/mpi/include/ops_mpi.hpp"
#include "kutergin_a_closest_pair/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kutergin_a_closest_pair {

class KuterginAClosestPairPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    const size_t vector_size = 100000000;
    std::vector<int> vector(vector_size);

    for (size_t i = 0; i < vector_size; i++) {
      vector[i] = static_cast<int>(i % 1000) + ((i & 1) == 0 ? 1 : -1);
    }
    input_data_ = std::move(vector);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(KuterginAClosestPairPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KuterginAClosestPairMPI, KuterginAClosestPairSEQ>(
    PPC_SETTINGS_kutergin_a_closest_pair);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KuterginAClosestPairPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KuterginAClosestPairPerfTests, kGtestValues, kPerfTestName);

}  // namespace kutergin_a_closest_pair
