#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "makovskiy_i_allreduce/common/include/common.hpp"
#include "makovskiy_i_allreduce/mpi/include/ops_mpi.hpp"
#include "makovskiy_i_allreduce/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace makovskiy_i_allreduce {

class AllreduceRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &input = std::get<0>(test_param);
    return "Size_" + std::to_string(input.size());
  }

 protected:
  InType GetTestInputData() final {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    return std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const auto &expected = std::get<1>(params);

    return !output_data.empty() && (output_data[0] == expected[0]);
  }
};

TEST_P(AllreduceRunFuncTests, SumVector) {
  ExecuteTest(GetParam());
}

namespace {

const auto kTestCases = std::array<TestType, 5>{
    TestType{InType{1, 2, 3, 4}, OutType{10}}, TestType{InType{10, -10, 5}, OutType{5}},
    TestType{InType{100}, OutType{100}}, TestType{InType{}, OutType{0}}, TestType{InType{1, 1, 1, 1, 1}, OutType{5}}};

const auto kFuncTestName = AllreduceRunFuncTests::PrintFuncTestName<AllreduceRunFuncTests>;

const auto kSeqTasks = ppc::util::AddFuncTask<TestTaskSEQ, InType>(kTestCases, PPC_SETTINGS_makovskiy_i_allreduce);
const auto kSeqGtestValues = ppc::util::ExpandToValues(kSeqTasks);

const auto kMpiTasks = ppc::util::AddFuncTask<TestTaskMPI, InType>(kTestCases, PPC_SETTINGS_makovskiy_i_allreduce);
const auto kMpiGtestValues = ppc::util::ExpandToValues(kMpiTasks);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(AllreducePerfSEQ, AllreduceRunFuncTests, kSeqGtestValues, kFuncTestName);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(AllreducePerfMPI, AllreduceRunFuncTests, kMpiGtestValues, kFuncTestName);

}  // namespace

}  // namespace makovskiy_i_allreduce
