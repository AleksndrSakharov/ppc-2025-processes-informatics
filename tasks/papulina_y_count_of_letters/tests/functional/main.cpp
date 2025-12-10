#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>

#include "papulina_y_count_of_letters/common/include/common.hpp"
#include "papulina_y_count_of_letters/mpi/include/ops_mpi.hpp"
#include "papulina_y_count_of_letters/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace papulina_y_count_of_letters {

class PapulinaYRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string s = std::string(std::get<0>(std::get<0>(params)));
    if (s != "generate") {
      input_data_ = std::string(std::get<0>(std::get<0>(params)));
      expectedResult_ = std::get<1>(std::get<0>(params));
    } else {
      std::string data = GenerateData(100);  // будет генерироваться строка, в которой буквенных символов ровно 100
      input_data_ = data;
      expectedResult_ = 100;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expectedResult_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expectedResult_ = 0;
  static std::string GenerateData(size_t count) {
    std::mt19937 gen(count);  // константный, чтобы у всех потоков была одна и та же строка при генерации
    std::uniform_int_distribution<size_t> length_dist(100, 500);
    std::uniform_int_distribution<size_t> char_dist(32, 126);

    size_t length = length_dist(gen);
    std::string generated_string;
    generated_string.reserve(length);

    std::string all_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::uniform_int_distribution<size_t> letter_dist(0, all_letters.size() - 1);
    // добавляем все буквы
    for (size_t i = 0; i < count; i++) {
      char c = all_letters[letter_dist(gen)];
      generated_string += c;
    }
    std::uniform_int_distribution<> non_letter_dist(48, 57);  // цифры 0-9
    for (size_t i = count; i < length; ++i) {
      char c = static_cast<char>(non_letter_dist(gen));
      generated_string += c;
    }
    std::shuffle(generated_string.begin(), generated_string.end(), gen);
    return generated_string;
  }
};

namespace {

TEST_P(PapulinaYRunFuncTestsProcesses, CountOfLetters) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 20> kTestParam = {
    std::make_tuple(std::make_tuple("", 0), "test1"),
    std::make_tuple(std::make_tuple("abcd", 4), "test2"),
    std::make_tuple(std::make_tuple("aabcd123abcd123abcd", 13), "test3"),
    std::make_tuple(std::make_tuple("abcd_____________123abcd", 8), "test4"),
    std::make_tuple(std::make_tuple("a", 1), "test5"),
    std::make_tuple(std::make_tuple("1243356", 0), "test6"),
    std::make_tuple(std::make_tuple("a1a1a1a1a1a1a1a1a1a1a1a1", 12), "test7"),
    std::make_tuple(std::make_tuple("!@345678&*()", 0), "test8"),
    std::make_tuple(std::make_tuple("aaaaaaaaaaaaaaaaaaaa", 20), "test9"),
    std::make_tuple(std::make_tuple("tatatatatatatatatatatatatatatatatatatatata", 42), "test10"),
    std::make_tuple(std::make_tuple("er11er11er11er11", 8), "test11"),
    std::make_tuple(std::make_tuple("eee___eee__", 6), "test12"),
    std::make_tuple(std::make_tuple("eee___eee__EEE", 9), "test13"),
    std::make_tuple(std::make_tuple("EEEE___EEEE", 8), "test14"),
    std::make_tuple(
        std::make_tuple(
            "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
            100),
        "test15"),
    std::make_tuple(std::make_tuple("фбсдуащуо", 0), "test16"),
    std::make_tuple(std::make_tuple("aa", 2), "test17"),
    std::make_tuple(std::make_tuple("aaa", 3), "test18"),
    std::make_tuple(std::make_tuple("aabb0123456789", 4), "test19"),
    std::make_tuple(std::make_tuple("generate", 4), "test20")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PapulinaYCountOfLettersMPI, InType>(kTestParam, PPC_SETTINGS_papulina_y_count_of_letters),
    ppc::util::AddFuncTask<PapulinaYCountOfLettersSEQ, InType>(kTestParam, PPC_SETTINGS_papulina_y_count_of_letters));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PapulinaYRunFuncTestsProcesses::PrintFuncTestName<PapulinaYRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, PapulinaYRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace papulina_y_count_of_letters
