#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

#include "gasenin_l_lex_dif/common/include/common.hpp"
#include "gasenin_l_lex_dif/mpi/include/ops_mpi.hpp"
#include "gasenin_l_lex_dif/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gasenin_l_lex_dif {

class GaseninLRunPerfTestsLexDif : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string long_str1;
    std::string long_str2;
    std::vector<int> lengths(2, 0);

    if (rank == 0) {
      long_str1 = std::string(100000000, 'a');
      long_str2 = std::string(100000000, 'a');
      long_str2[5000000] = 'b';

      lengths[0] = static_cast<int>(long_str1.length());
      lengths[1] = static_cast<int>(long_str2.length());
    }

    MPI_Bcast(lengths.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
      long_str1.resize(static_cast<size_t>(lengths[0]));
      long_str2.resize(static_cast<size_t>(lengths[1]));
    }

    if (lengths[0] > 0) {
      MPI_Bcast(long_str1.data(), static_cast<int>(lengths[0]), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    if (lengths[1] > 0) {
      MPI_Bcast(long_str2.data(), static_cast<int>(lengths[1]), MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    input_data_ = {long_str1, long_str2};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == -1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GaseninLRunPerfTestsLexDif, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaseninLLexDifMPI, GaseninLLexDifSEQ>(PPC_SETTINGS_gasenin_l_lex_dif);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GaseninLRunPerfTestsLexDif::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(LexDifPerfTests, GaseninLRunPerfTestsLexDif, kGtestValues, kPerfTestName);

}  // namespace gasenin_l_lex_dif
