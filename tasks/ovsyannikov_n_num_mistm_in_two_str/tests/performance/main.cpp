#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "ovsyannikov_n_num_mistm_in_two_str/common/include/common.hpp"
#include "ovsyannikov_n_num_mistm_in_two_str/mpi/include/ops_mpi.hpp"
#include "ovsyannikov_n_num_mistm_in_two_str/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ovsyannikov_n_num_mistm_in_two_str {

class OvsyannikovNRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    // Размер 100 миллионов символов
    const int benchmark_size = 100'000'000;

    std::string sample_a(benchmark_size, 'a');
    std::string sample_b(benchmark_size, 'a');

    expected_output_ = 0;
    for (int i = 0; i < benchmark_size; ++i) {
      if (i % 2 == 0) {
        sample_b[i] = 'b';
        expected_output_++;
      }
    }

    input_data_ = std::make_pair(sample_a, sample_b);
  }

  bool CheckTestOutputData(OutType &calculated_res) final {
    return calculated_res == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_ = 0;
};
;

TEST_P(OvsyannikovNRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OvsyannikovNNumMistmInTwoStrMPI, OvsyannikovNNumMistmInTwoStrSEQ>(
        PPC_SETTINGS_ovsyannikov_n_num_mistm_in_two_str);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OvsyannikovNRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, OvsyannikovNRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace ovsyannikov_n_num_mistm_in_two_str
