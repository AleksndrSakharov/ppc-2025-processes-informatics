#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>
#include <tuple>

#include "telnov_transfer_one_all/common/include/common.hpp"
#include "telnov_transfer_one_all/mpi/include/ops_mpi.hpp"
#include "telnov_transfer_one_all/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace telnov_transfer_one_all {

class TelnovTransferOneAllFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const int array_size = 10;
    input_data_.resize(array_size);

    for (int i = 0; i < array_size; ++i) {
      input_data_[i] = input_data_[i] = (i * 13) % 100;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::equal(input_data_.begin(), input_data_.end(), output_data.begin());
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(TelnovTransferOneAllFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TelnovTransferOneAllMPI<int>, InType>(kTestParam, PPC_SETTINGS_telnov_transfer_one_all),
    ppc::util::AddFuncTask<TelnovTransferOneAllSEQ<int>, InType>(kTestParam, PPC_SETTINGS_telnov_transfer_one_all));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TelnovTransferOneAllFuncTests::PrintFuncTestName<TelnovTransferOneAllFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TelnovTransferOneAllFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace telnov_transfer_one_all
