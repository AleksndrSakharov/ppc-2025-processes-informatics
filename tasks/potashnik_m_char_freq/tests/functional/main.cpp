#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "potashnik_m_char_freq/common/include/common.hpp"
#include "potashnik_m_char_freq/mpi/include/ops_mpi.hpp"
#include "potashnik_m_char_freq/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace potashnik_m_char_freq {

class PotashnikMCharFreqFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string str;
    char chr = 0;

    int seed = params;

    // Generating character
    chr = static_cast<char>('a' + (seed % 26));

    // Generating string
    for (int i = 0; i < params; i++) {
      char c = static_cast<char>('a' + ((i * 7 + 13 + seed / 2) % 26));
      str += c;
    }

    input_data_ = std::make_tuple(str, chr);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int res = 0;
    std::string str = std::get<0>(input_data_);
    char chr = std::get<1>(input_data_);

    int string_size = static_cast<int>(str.size());
    for (int i = 0; i < string_size; i++) {
      if (str[i] == chr) {
        res++;
      }
    }

    return (res == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(PotashnikMCharFreqFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {1, 5, 10, 20, 100, 1000, 2000, 5000, 10000, 20000};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PotashnikMCharFreqMPI, InType>(kTestParam, PPC_SETTINGS_potashnik_m_char_freq),
    ppc::util::AddFuncTask<PotashnikMCharFreqSEQ, InType>(kTestParam, PPC_SETTINGS_potashnik_m_char_freq));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PotashnikMCharFreqFuncTests::PrintFuncTestName<PotashnikMCharFreqFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, PotashnikMCharFreqFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace potashnik_m_char_freq
