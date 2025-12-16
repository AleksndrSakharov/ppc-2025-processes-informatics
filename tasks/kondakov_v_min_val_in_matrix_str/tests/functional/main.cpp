#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>

#include "kondakov_v_min_val_in_matrix_str/common/include/common.hpp"
#include "kondakov_v_min_val_in_matrix_str/mpi/include/ops_mpi.hpp"
#include "kondakov_v_min_val_in_matrix_str/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kondakov_v_min_val_in_matrix_str {

class KondakovVMinValMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    if (test_name == "small") {
      input_data_ = {{3, 1, 4}, {1, 5, 9}, {2, 6, 5}};
      expected_output_ = {1, 1, 2};
    } else if (test_name == "single_row") {
      input_data_ = {{-10, 0, 10}};
      expected_output_ = {-10};
    } else if (test_name == "negative") {
      input_data_ = {{-5, -2, -8}, {-1, -10, -3}};
      expected_output_ = {-8, -10};
    } else if (test_name == "one_element") {
      input_data_ = {{42}};
      expected_output_ = {42};
    } else if (test_name == "empty") {
      input_data_ = {};
      expected_output_ = {};
    } else {
      throw std::runtime_error("Unknown test case: " + test_name);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(KondakovVMinValMatrixFuncTests, FindMinInRows) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {"small", "single_row", "negative", "one_element", "empty"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KondakovVMinValMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_kondakov_v_min_val_in_matrix_str),
    ppc::util::AddFuncTask<KondakovVMinValMatrixMPI, InType>(kTestParam,
                                                             PPC_SETTINGS_kondakov_v_min_val_in_matrix_str));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KondakovVMinValMatrixFuncTests::PrintFuncTestName<KondakovVMinValMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(MinValMatrixTests, KondakovVMinValMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kondakov_v_min_val_in_matrix_str
