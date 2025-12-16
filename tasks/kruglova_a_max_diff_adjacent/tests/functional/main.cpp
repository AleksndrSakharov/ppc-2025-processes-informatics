#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "kruglova_a_max_diff_adjacent/common/include/common.hpp"
#include "kruglova_a_max_diff_adjacent/mpi/include/ops_mpi.hpp"
#include "kruglova_a_max_diff_adjacent/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kruglova_a_max_diff_adjacent {

class KruglovaAMaxDiffAdjacentTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);
    input_data_.resize(size);

    float acc = 0.0F;
    for (int i = 0; i < size; ++i) {
      float step = (static_cast<float>((i % 5) - 2) * 0.7F) + (static_cast<float>((i % 3) - 1) * 0.3F);
      acc += step;
      input_data_[i] = acc;
    }

    expected_output_ = 0.0F;
    if (input_data_.size() > 1) {
      float max_diff = 0.0F;
      for (size_t i = 0; i + 1 < input_data_.size(); ++i) {
        float diff = std::abs(input_data_[i + 1] - input_data_[i]);
        max_diff = std::max(diff, max_diff);
      }
      expected_output_ = max_diff;
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
  float expected_output_ = 0.0F;
};

namespace {

TEST_P(KruglovaAMaxDiffAdjacentTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7"),
                                            std::make_tuple(1, "1")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KruglovaAMaxDiffAdjacentMPI, InType>(kTestParam, PPC_SETTINGS_kruglova_a_max_diff_adjacent),
    ppc::util::AddFuncTask<KruglovaAMaxDiffAdjacentSEQ, InType>(kTestParam, PPC_SETTINGS_kruglova_a_max_diff_adjacent));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KruglovaAMaxDiffAdjacentTests::PrintFuncTestName<KruglovaAMaxDiffAdjacentTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KruglovaAMaxDiffAdjacentTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kruglova_a_max_diff_adjacent
