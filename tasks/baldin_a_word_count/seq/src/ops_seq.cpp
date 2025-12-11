#include "baldin_a_word_count/seq/include/ops_seq.hpp"

#include <cctype>
#include <cstddef>
#include <string>

#include "baldin_a_word_count/common/include/common.hpp"

namespace baldin_a_word_count {

namespace {

bool IsWordChar(char c) {
  return ((std::isalnum(static_cast<unsigned char>(c)) != 0) || c == '-' || c == '_');
}

}  // namespace

BaldinAWordCountSEQ::BaldinAWordCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BaldinAWordCountSEQ::ValidationImpl() {
  return true;
}

bool BaldinAWordCountSEQ::PreProcessingImpl() {
  return true;
}

bool BaldinAWordCountSEQ::RunImpl() {
  std::string &input = GetInput();
  size_t count = 0;
  bool in_word = false;
  for (char c : input) {
    if (IsWordChar(c)) {
      if (!in_word) {
        in_word = true;
        count++;
      }
    } else {
      in_word = false;
    }
  }

  GetOutput() = count;
  return true;
}

bool BaldinAWordCountSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_word_count
