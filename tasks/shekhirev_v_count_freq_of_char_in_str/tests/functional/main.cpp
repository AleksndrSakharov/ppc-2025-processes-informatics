#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <tuple>

#include "../../common/include/common.hpp"
#include "../../mpi/include/ops_mpi.hpp"
#include "../../seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shekhirev_v_char_freq_seq {

TEST(ShekhirevVCharFreqCommon, InputDataStruct) {
  InputData d1("test", 't');
  InputData d2("test", 't');
  InputData d3("diff", 'd');

  ASSERT_TRUE(d1 == d2);
  ASSERT_FALSE(d1 == d3);

  std::stringstream ss;
  ss << d1;
  ASSERT_FALSE(ss.str().empty());
}

class ShekhirevVCharFreqFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int process_rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    }
    if (process_rank == 0) {
      return output_data == expected_output_;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_ = 0;
};

namespace {

template <typename T>
void RunPipelineStep1(const T &task) {
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
}

template <typename T>
void RunPipelineStep2(const T &task) {
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
}

TEST_P(ShekhirevVCharFreqFuncTests, Test) {
  auto test_param = GetParam();
  auto task_getter = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTaskGetter)>(test_param);
  const auto &test_case_params =
      std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(test_param);
  const auto &input_data = std::get<0>(test_case_params);

  auto task = task_getter(input_data);

  RunPipelineStep1(task);
  RunPipelineStep2(task);

  ASSERT_TRUE(CheckTestOutputData(task->GetOutput()));
}

const std::array<TestType, 6> kTestCases = {std::make_tuple(InputData("hello world", 'l'), 3, "simple_word"),
                                            std::make_tuple(InputData("hello world", 'z'), 0, "not_found"),
                                            std::make_tuple(InputData("", 'a'), 0, "empty_string"),
                                            std::make_tuple(InputData("aaaaa", 'a'), 5, "all_target"),
                                            std::make_tuple(InputData("abacabadabacaba", 'a'), 8, "longer_string"),
                                            std::make_tuple(InputData("zaaaaaaz", 'z'), 2, "edges")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<CharFreqSequential, InType>(kTestCases, PPC_SETTINGS_shekhirev_v_count_freq_of_char_in_str),
    ppc::util::AddFuncTask<shekhirev_v_char_freq_mpi::CharFreqMPI, InType>(
        kTestCases, PPC_SETTINGS_shekhirev_v_count_freq_of_char_in_str));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kTestName = ShekhirevVCharFreqFuncTests::PrintFuncTestName<ShekhirevVCharFreqFuncTests>;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(CharFreqTests, ShekhirevVCharFreqFuncTests, kGtestValues, kTestName);

}  // namespace

}  // namespace shekhirev_v_char_freq_seq
