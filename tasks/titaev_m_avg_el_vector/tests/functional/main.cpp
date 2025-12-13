#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>

#include "titaev_m_avg_el_vector/common/include/common.hpp"
#include "titaev_m_avg_el_vector/mpi/include/ops_mpi.hpp"
#include "titaev_m_avg_el_vector/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace titaev_m_avg_el_vector {

class TitaevMElemVecAvgFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::to_string(std::get<0>(param)) + "_" + std::get<1>(param);
  }

 protected:
  void SetUp() override {
    int len = std::get<0>(std::get<2>(GetParam()));
    input_data_.assign(len, 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data - 1.0) < 1e-12;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(TitaevMElemVecAvgFuncTest, RunTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kParams = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TitaevMElemVecsAvgMPI, InType>(kParams, PPC_SETTINGS_titaev_m_avg_el_vector),
                   ppc::util::AddFuncTask<TitaevMElemVecsAvgSEQ, InType>(kParams, PPC_SETTINGS_titaev_m_avg_el_vector));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TitaevMElemVecAvgFuncTest::PrintFuncTestName<TitaevMElemVecAvgFuncTest>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TitaevMElemVecAvgFuncTest, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace titaev_m_avg_el_vector
