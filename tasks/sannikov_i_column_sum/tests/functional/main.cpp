#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "sannikov_i_column_sum/common/include/common.hpp"
#include "sannikov_i_column_sum/mpi/include/ops_mpi.hpp"
#include "sannikov_i_column_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sannikov_i_column_sum {

class SannikovIColumnSumFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    OutType sums_vec(input_data_.front().size(), 0);
    for (std::size_t i = 0; i < (input_data_.size()); i++) {
      for (std::size_t j = 0; j < (input_data_[i].size()); j++) {
        sums_vec[j] += input_data_[i][j];
      }
    }
    return (sums_vec == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(SannikovIColumnSumFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 16> kTestParam = {
    std::make_tuple(std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, "matrix3x3"),
    std::make_tuple(std::vector<std::vector<int>>{{-1, -2, -3}, {-4, -5, -6}, {-7, -8, -9}}, "matrix3x3negative"),
    std::make_tuple(std::vector<std::vector<int>>{{1, 2}, {1, 2}, {1, 2}, {1, 2}}, "matrix4x2"),
    std::make_tuple(std::vector<std::vector<int>>{{0}}, "matrix1x1zero"),
    std::make_tuple(std::vector<std::vector<int>>{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, "matrix3x3zero"),
    std::make_tuple(std::vector<std::vector<int>>{{1, 10, 15, 19, 90, 3}}, "matrix1x6"),
    std::make_tuple(std::vector<std::vector<int>>{{1037, 2385, 8543}, {286, 629, 1094}, {8306, 6290, 375}},
                    "matrix3x3big_nums"),
    std::make_tuple(std::vector<std::vector<int>>{{1037, -2385, 8543}, {-286, 629, -1094}, {8306, 6290, -375}},
                    "matrix3x3big_nums_with_negative"),
    std::make_tuple(std::vector<std::vector<int>>{{-5, 0, 7, -3, 10}}, "matrix1x5_mix"),
    std::make_tuple(std::vector<std::vector<int>>{{1}, {2}, {-3}, {4}, {-5}}, "matrix5x1_column"),
    std::make_tuple(std::vector<std::vector<int>>{{0, -1, 2, -3}, {4, 0, -5, 6}}, "matrix2x4_mix"),
    std::make_tuple(std::vector<std::vector<int>>{{1, 2, 3, 4, 5}, {-1, 0, 7, -2, 3}, {10, -5, 8, 0, -4}},
                    "matrix3x5_mix"),
    std::make_tuple(std::vector<std::vector<int>>{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}, "matrix1x10_wide"),
    std::make_tuple(std::vector<std::vector<int>>{{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}}, "matrix10x1_tall"),
    std::make_tuple(std::vector<std::vector<int>>{{5, 5}, {5, 5}}, "matrix2x2_same_values"),
    std::make_tuple(std::vector<std::vector<int>>{{100000, -50000, 30000}, {20000, -10000, 5000}, {1, 2, 3}},
                    "matrix3x3_large_positive_mixed")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SannikovIColumnSumMPI, InType>(kTestParam, PPC_SETTINGS_sannikov_i_column_sum),
    ppc::util::AddFuncTask<SannikovIColumnSumSEQ, InType>(kTestParam, PPC_SETTINGS_sannikov_i_column_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SannikovIColumnSumFuncTests::PrintFuncTestName<SannikovIColumnSumFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SannikovIColumnSumFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sannikov_i_column_sum
