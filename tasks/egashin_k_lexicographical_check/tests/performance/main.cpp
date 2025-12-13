#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>

#include "egashin_k_lexicographical_check/common/include/common.hpp"
#include "egashin_k_lexicographical_check/mpi/include/ops_mpi.hpp"
#include "egashin_k_lexicographical_check/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace egashin_k_lexicographical_check {

class EgashinKLexCheckPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    size_t size = 20000000;
    std::string s1(size, 'a');
    std::string s2(size, 'a');
    s2.back() = 'b';
    input_ = {s1, s2};
    expected_ = true;
  }

  bool CheckTestOutputData(OutType &output) override {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    return output == expected_;
  }

  InType GetTestInputData() override {
    return input_;
  }

 private:
  InType input_;
  OutType expected_ = false;
};

TEST_P(EgashinKLexCheckPerfTest, Performance) {
  ExecuteTest(GetParam());
}

const auto kPerfParams =
    ppc::util::MakeAllPerfTasks<InType, TestTaskMPI, TestTaskSEQ>(PPC_SETTINGS_egashin_k_lexicographical_check);

INSTANTIATE_TEST_SUITE_P(EgashinKLexCheckPerf, EgashinKLexCheckPerfTest, ppc::util::TupleToGTestValues(kPerfParams));

}  // namespace egashin_k_lexicographical_check
