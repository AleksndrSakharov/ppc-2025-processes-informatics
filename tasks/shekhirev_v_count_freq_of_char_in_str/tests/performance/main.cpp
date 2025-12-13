#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>

#include "../../common/include/common.hpp"
#include "../../mpi/include/ops_mpi.hpp"
#include "../../seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace shekhirev_v_char_freq_seq {

class ShekhirevVCharFreqPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      const size_t size = 1000000000;
      input_data_.str.resize(size, 'a');
      input_data_.str[size / 2] = 'b';
      input_data_.str[size - 1] = 'b';
      input_data_.target = 'b';
    } else {
      input_data_.target = 'b';
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int process_rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    }
    if (process_rank == 0) {
      return output_data == 2;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(ShekhirevVCharFreqPerfTests, PerfTest) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, shekhirev_v_char_freq_mpi::CharFreqMPI, CharFreqSequential>(
        PPC_SETTINGS_shekhirev_v_count_freq_of_char_in_str);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(CharFreqPerf, ShekhirevVCharFreqPerfTests, ppc::util::TupleToGTestValues(kAllPerfTasks),
                         ShekhirevVCharFreqPerfTests::CustomPerfTestName);

}  // namespace

}  // namespace shekhirev_v_char_freq_seq
