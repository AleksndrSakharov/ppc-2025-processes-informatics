#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kutuzov_i_elem_vec_average/common/include/common.hpp"
#include "kutuzov_i_elem_vec_average/mpi/include/ops_mpi.hpp"
#include "kutuzov_i_elem_vec_average/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kutuzov_i_elem_vec_average {

class KutuzovIElemVecAverageFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::vector<double>(param, 0.0);

    answer_ = 0.0;
    int num = static_cast<int>(param);
    for (int i = 0; i < num; i++) {
      double value = static_cast<double>(i * i) - (static_cast<double>(param) / 2.0);
      input_data_[i] = value;
      answer_ += value;
    }
    answer_ /= static_cast<double>(param);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool result = abs(output_data - answer_) < 0.0001;

    return result;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  double answer_ = 0.0;
};

namespace {

TEST_P(KutuzovIElemVecAverageFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {1, 10, 1000, 10000};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KutuzovIElemVecAverageMPI, InType>(kTestParam, PPC_SETTINGS_kutuzov_i_elem_vec_average),
    ppc::util::AddFuncTask<KutuzovIElemVecAverageSEQ, InType>(kTestParam, PPC_SETTINGS_kutuzov_i_elem_vec_average));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KutuzovIElemVecAverageFuncTests::PrintFuncTestName<KutuzovIElemVecAverageFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KutuzovIElemVecAverageFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kutuzov_i_elem_vec_average
