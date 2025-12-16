#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "akimov_i_words_string_count/common/include/common.hpp"
#include "akimov_i_words_string_count/mpi/include/ops_mpi.hpp"
#include "akimov_i_words_string_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace akimov_i_words_string_count {

class AkimovIWordsStringCountFromFileFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_akimov_i_words_string_count, "words.txt");

    std::ifstream file(abs_path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open words.txt at path: " + abs_path);
    }

    std::string content;
    {
      std::ostringstream ss;
      ss << file.rdbuf();
      content = ss.str();
    }
    file.close();

    for (char &c : content) {
      if (c == '\n' || c == '\r' || c == '\t') {
        c = ' ';
      }
    }

    input_data_ = InType(content.begin(), content.end());

    is_valid_ = !content.empty();

    expected_result_ = 0;
    if (is_valid_) {
      bool in_word = false;
      for (char c : content) {
        if (c != ' ' && !in_word) {
          in_word = true;
          ++expected_result_;
        } else if (c == ' ' && in_word) {
          in_word = false;
        }
      }
    }

    (void)std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int mpi_initialized = 0;

    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    } else {
      rank = 0;
    }

    if (!is_valid_ || rank != 0) {
      return true;
    }
    return output_data == expected_result_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0;
  bool is_valid_ = true;
};

namespace {

TEST_P(AkimovIWordsStringCountFromFileFuncTests, CountWordsFromFile) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {std::make_tuple(0, std::string("default"))};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<AkimovIWordsStringCountMPI, InType>(kTestParam, PPC_SETTINGS_akimov_i_words_string_count),
    ppc::util::AddFuncTask<AkimovIWordsStringCountSEQ, InType>(kTestParam, PPC_SETTINGS_akimov_i_words_string_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    AkimovIWordsStringCountFromFileFuncTests::PrintFuncTestName<AkimovIWordsStringCountFromFileFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, AkimovIWordsStringCountFromFileFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace akimov_i_words_string_count
