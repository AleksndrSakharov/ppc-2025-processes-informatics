#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "telnov_counting_the_frequency/common/include/common.hpp"
#include "telnov_counting_the_frequency/mpi/include/ops_mpi.hpp"
#include "telnov_counting_the_frequency/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace telnov_counting_the_frequency {

class TelnovCountingTheFrequencyFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    // Берём параметр теста
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);

    // Готовим глобальную строку
    telnov_counting_the_frequency::GlobalData::g_data_string.clear();
    telnov_counting_the_frequency::GlobalData::g_data_string.resize(2'000'000, 'a');

    // Пишем X в начале строки
    for (int i = 0; i < input_data_; i++) {
      telnov_counting_the_frequency::GlobalData::g_data_string[i] = 'X';
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(TelnovCountingTheFrequencyFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TelnovCountingTheFrequencyMPI, InType>(
                                               kTestParam, PPC_SETTINGS_telnov_counting_the_frequency),
                                           ppc::util::AddFuncTask<TelnovCountingTheFrequencySEQ, InType>(
                                               kTestParam, PPC_SETTINGS_telnov_counting_the_frequency));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    TelnovCountingTheFrequencyFuncTestsProcesses::PrintFuncTestName<TelnovCountingTheFrequencyFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TelnovCountingTheFrequencyFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace telnov_counting_the_frequency
