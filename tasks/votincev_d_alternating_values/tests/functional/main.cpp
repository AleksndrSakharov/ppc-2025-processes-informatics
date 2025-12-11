#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>  // для size_t
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "votincev_d_alternating_values/common/include/common.hpp"
#include "votincev_d_alternating_values/mpi/include/ops_mpi.hpp"
#include "votincev_d_alternating_values/seq/include/ops_seq.hpp"

namespace votincev_d_alternating_values {

class VotincevDAlternatigValuesRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  // считываем/генерируем данные
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string input_data_source =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_votincev_d_alternating_values, param + ".txt");

    std::ifstream file(input_data_source);
    int expect_res = 0;
    file >> expect_res;  // считываю предполагаемый ответ

    expected_res_ = expect_res;  // устанавливаю его

    int data_count = 0;
    file >> data_count;  // получаю количество элементов

    std::vector<double> vect_data;
    for (int i = 0; i < data_count; i++) {
      double elem = 0.0;
      file >> elem;
      vect_data.push_back(elem);
    }

    input_data_ = vect_data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // процессы (кроме 0-го) не сверяются с ответом
    if (output_data == -1) {
      return true;
    }

    // 0й процесс должен иметь правильный результат
    return output_data == expected_res_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_res_ = -1;
};

namespace {

TEST_P(VotincevDAlternatigValuesRunFuncTestsProcesses, CountSwapsFromGenerator) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {"test1", "test2", "test3", "test4", "test5",
                                             "test6", "test7", "test8", "test9", "test10"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<VotincevDAlternatingValuesMPI, InType>(
                                               kTestParam, PPC_SETTINGS_votincev_d_alternating_values),
                                           ppc::util::AddFuncTask<VotincevDAlternatingValuesSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_votincev_d_alternating_values));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    VotincevDAlternatigValuesRunFuncTestsProcesses::PrintFuncTestName<VotincevDAlternatigValuesRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(CountSwapsFromGeneratorr, VotincevDAlternatigValuesRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace votincev_d_alternating_values
