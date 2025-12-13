#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

#include "titaev_m_avg_el_vector/common/include/common.hpp"
#include "titaev_m_avg_el_vector/mpi/include/ops_mpi.hpp"
#include "titaev_m_avg_el_vector/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace titaev_m_avg_el_vector {

class TitaevMElemVecAvgPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr std::size_t kCountDefault = 50000000;

  TitaevMElemVecAvgPerfTest() = default;

 protected:
  void SetUp() override {
    input_data_.assign(kCountDefault, 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data - 1.0) < 1e-12;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(TitaevMElemVecAvgPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TitaevMElemVecsAvgMPI, TitaevMElemVecsAvgSEQ>(
    PPC_SETTINGS_titaev_m_avg_el_vector);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TitaevMElemVecAvgPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TitaevMElemVecAvgPerfTest, kGtestValues, kPerfTestName);

}  // namespace titaev_m_avg_el_vector
