#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lukin_i_cannon_algorithm/common/include/common.hpp"
#include "lukin_i_cannon_algorithm/mpi/include/ops_mpi.hpp"
#include "lukin_i_cannon_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace lukin_i_cannon_algorithm {
const double kEpsilon = 1e-9;

class LukinIRunFuncTestsProcesses2 : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_lukin_i_cannon_algorithm, params + ".txt");

    std::ifstream ifstr(path);

    size_ = 0;
    ifstr >> size_;

    std::vector<double> a(static_cast<size_t>(size_ * size_));
    std::vector<double> b(static_cast<size_t>(size_ * size_));
    std::vector<double> c(static_cast<size_t>(size_ * size_), 0);

    for (int i = 0; i < size_ * size_; i++) {
      ifstr >> a[i];
    }

    for (int i = 0; i < size_ * size_; i++) {
      ifstr >> b[i];
    }

    std::vector<double> a_copy = a;
    std::vector<double> b_copy = b;

    input_data_ = std::make_tuple(std::move(a_copy), std::move(b_copy), size_);

    for (int i = 0; i < size_; i++) {
      for (int k = 0; k < size_; k++) {
        double fixed = a[(i * size_) + k];
        for (int j = 0; j < size_; j++) {
          c[(i * size_) + j] += fixed * b[(k * size_) + j];
        }
      }
    }

    expected_data_ = std::move(c);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const int size = static_cast<int>(expected_data_.size());
    for (int i = 0; i < size; i++) {
      if (std::abs(expected_data_[i] - output_data[i]) > kEpsilon) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;

  OutType expected_data_;

  int size_ = 0;
};

namespace {

TEST_P(LukinIRunFuncTestsProcesses2, Cannon) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
    "matrix_2",
    "matrix_4",
    "matrix_8",
    "matrix_12",
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<LukinICannonAlgorithmMPI, InType>(kTestParam, PPC_SETTINGS_lukin_i_cannon_algorithm),
    ppc::util::AddFuncTask<LukinICannonAlgorithmSEQ, InType>(kTestParam, PPC_SETTINGS_lukin_i_cannon_algorithm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LukinIRunFuncTestsProcesses2::PrintFuncTestName<LukinIRunFuncTestsProcesses2>;

INSTANTIATE_TEST_SUITE_P(CannonAlgorithmTest, LukinIRunFuncTestsProcesses2, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace lukin_i_cannon_algorithm
