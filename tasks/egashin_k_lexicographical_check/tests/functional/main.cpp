#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include "egashin_k_lexicographical_check/common/include/common.hpp"
#include "egashin_k_lexicographical_check/mpi/include/ops_mpi.hpp"
#include "egashin_k_lexicographical_check/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace egashin_k_lexicographical_check {

class EgashinKLexCheckFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_ = std::get<0>(param);
    expected_ = std::get<1>(param);
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

namespace {

TEST_P(EgashinKLexCheckFuncTest, LexicographicalOrder) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTests = {
    std::make_tuple(std::make_pair("abc", "abd"), true),
    std::make_tuple(std::make_pair("abd", "abc"), false),
    std::make_tuple(std::make_pair("abc", "abc"), false),
    std::make_tuple(std::make_pair("", "a"), true),
    std::make_tuple(std::make_pair("a", ""), false),
    std::make_tuple(std::make_pair("apple", "apricot"), true),
    std::make_tuple(std::make_pair("short", "shorter"), true),
    std::make_tuple(std::make_pair("shorter", "short"), false),
    std::make_tuple(std::make_pair("123", "1234"), true),
    std::make_tuple(std::make_pair(std::string(100, 'a'), std::string(100, 'b')), true)};

const auto kTaskParams =
    std::tuple_cat(ppc::util::AddFuncTask<TestTaskSEQ, InType>(kTests, PPC_SETTINGS_egashin_k_lexicographical_check),
                   ppc::util::AddFuncTask<TestTaskMPI, InType>(kTests, PPC_SETTINGS_egashin_k_lexicographical_check));

INSTANTIATE_TEST_SUITE_P(EgashinKLexCheckFunc, EgashinKLexCheckFuncTest, ppc::util::ExpandToValues(kTaskParams));

}  // namespace

}  // namespace egashin_k_lexicographical_check
