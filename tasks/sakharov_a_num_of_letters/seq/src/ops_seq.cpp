#include "sakharov_a_num_of_letters/seq/include/ops_seq.hpp"

#include <cctype>
#include <utility>

#include "sakharov_a_num_of_letters/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sakharov_a_num_of_letters {

SakharovANumberOfLettersSEQ::SakharovANumberOfLettersSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SakharovANumberOfLettersSEQ::ValidationImpl() {
  return std::cmp_equal(std::get<1>(GetInput()).size(), std::get<0>(GetInput()));
}

bool SakharovANumberOfLettersSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SakharovANumberOfLettersSEQ::RunImpl() {
  const std::string &string_of_letters = std::get<1>(GetInput());
  int number_of_letter = 0;
  for (char c : string_of_letters) {
    if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
      number_of_letter++;
    }
  }
  GetOutput() = number_of_letter;
  return true;
}

bool SakharovANumberOfLettersSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sakharov_a_num_of_letters
