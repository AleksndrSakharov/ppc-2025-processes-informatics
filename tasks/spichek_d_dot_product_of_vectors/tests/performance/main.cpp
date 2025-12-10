#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"
#include "spichek_d_dot_product_of_vectors/mpi/include/ops_mpi.hpp"
#include "spichek_d_dot_product_of_vectors/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace spichek_d_dot_product_of_vectors {

class SpichekDDotProductOfVectorsRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    const size_t vector_size = 100000000;

    std::vector<int> vector1(vector_size);
    std::vector<int> vector2(vector_size);

    for (size_t i = 0; i < vector_size; ++i) {
      vector1[i] = static_cast<int>(i % 1000) + 1;
      vector2[i] = static_cast<int>((i * 2) % 1000) + 1;
    }

    input_data_ = std::make_pair(vector1, vector2);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data != 0);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SpichekDDotProductOfVectorsRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SpichekDDotProductOfVectorsMPI, SpichekDDotProductOfVectorsSEQ>(
        PPC_SETTINGS_spichek_d_dot_product_of_vectors);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SpichekDDotProductOfVectorsRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SpichekDDotProductOfVectorsRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace spichek_d_dot_product_of_vectors
