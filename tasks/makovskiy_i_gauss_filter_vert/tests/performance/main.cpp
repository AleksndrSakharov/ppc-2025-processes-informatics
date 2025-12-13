#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "makovskiy_i_gauss_filter_vert/common/include/common.hpp"
#include "makovskiy_i_gauss_filter_vert/mpi/include/ops_mpi.hpp"
#include "makovskiy_i_gauss_filter_vert/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace makovskiy_i_gauss_filter_vert {

class GaussFilterPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType GetTestInputData() final {
    constexpr int kWidth = 2000;
    constexpr int kHeight = 2000;
    std::vector<int> input_image(static_cast<size_t>(kWidth) * kHeight);
    for (size_t i = 0; i < input_image.size(); ++i) {
      input_image[i] = static_cast<int>(i % 256);
    }
    return std::make_tuple(std::move(input_image), kWidth, kHeight);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = ppc::util::GetMPIRank();
    if (rank == 0) {
      const auto &[_, width, height] = GetTestInputData();
      return output_data.size() == static_cast<size_t>(width) * height;
    }
    return true;
  }
};

TEST_P(GaussFilterPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {
const auto kPerfTestName = GaussFilterPerfTests::CustomPerfTestName;

const auto kSeqPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaussFilterSEQ>(PPC_SETTINGS_makovskiy_i_gauss_filter_vert);
const auto kSeqGtestValues = ppc::util::TupleToGTestValues(kSeqPerfTasks);

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, misc-use-anonymous-namespace, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(GaussFilterPerfSEQ, GaussFilterPerfTests, kSeqGtestValues, kPerfTestName);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, misc-use-anonymous-namespace, modernize-type-traits)

const auto kMpiPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaussFilterMPI>(PPC_SETTINGS_makovskiy_i_gauss_filter_vert);
const auto kMpiGtestValues = ppc::util::TupleToGTestValues(kMpiPerfTasks);

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, misc-use-anonymous-namespace, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(GaussFilterPerfMPI, GaussFilterPerfTests, kMpiGtestValues, kPerfTestName);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, misc-use-anonymous-namespace, modernize-type-traits)

}  // namespace
}  // namespace makovskiy_i_gauss_filter_vert
