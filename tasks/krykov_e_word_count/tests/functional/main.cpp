#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <string>
#include <tuple>

#include "krykov_e_word_count/common/include/common.hpp"
#include "krykov_e_word_count/mpi/include/ops_mpi.hpp"
#include "krykov_e_word_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace krykov_e_word_count {

class KrykovEWordCountFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  KrykovEWordCountFuncTests() = default;

  static std::string PrintTestParam(const TestType &test_param) {
    std::string text = std::get<0>(test_param);

    if (text.size() > 20) {
      text = text.substr(0, 20);
    }

    for (char &ch : text) {
      if ((std::isalnum(static_cast<unsigned char>(ch)) == 0) && ch != '_') {
        ch = '_';
      }
    }
    return text + "_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_{};
};

namespace {

TEST_P(KrykovEWordCountFuncTests, WordCountTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 22> kTestParam = {
    std::make_tuple(std::string("Hello world"), 2),
    std::make_tuple(std::string("One two  three   four"), 4),

    std::make_tuple(std::string("Single"), 1),
    std::make_tuple(std::string("  Leading"), 1),
    std::make_tuple(std::string("Trailing  "), 1),
    std::make_tuple(std::string("  Both  "), 1),

    std::make_tuple(std::string("Word1\tWord2\nWord3"), 3),
    std::make_tuple(std::string("Tab\t\t\tTabs"), 2),
    std::make_tuple(std::string("Line1\nLine2\nLine3"), 3),
    std::make_tuple(std::string("Mixed\t \nSpaces"), 2),

    std::make_tuple(std::string("a b c d e f"), 6),
    std::make_tuple(std::string("multiple     spaces"), 2),
    std::make_tuple(std::string("one\ntwo\nthree\nfour"), 4),

    std::make_tuple(std::string("A"), 1),
    std::make_tuple(std::string(" A "), 1),
    std::make_tuple(std::string("  A  B  "), 2),
    std::make_tuple(std::string("VeryLongWordWithoutSpaces"), 1),

    std::make_tuple(std::string("word-with-dash"), 1),
    std::make_tuple(std::string("under_score_word"), 1),
    std::make_tuple(std::string("mixed-dash_and_underscore"), 1),

    std::make_tuple(std::string("split test"), 2),
    std::make_tuple(std::string("abc def ghi jkl"), 4),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KrykovEWordCountMPI, InType>(kTestParam, PPC_SETTINGS_krykov_e_word_count),
                   ppc::util::AddFuncTask<KrykovEWordCountSEQ, InType>(kTestParam, PPC_SETTINGS_krykov_e_word_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KrykovEWordCountFuncTests::PrintFuncTestName<KrykovEWordCountFuncTests>;

INSTANTIATE_TEST_SUITE_P(WordCountTests, KrykovEWordCountFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace krykov_e_word_count
