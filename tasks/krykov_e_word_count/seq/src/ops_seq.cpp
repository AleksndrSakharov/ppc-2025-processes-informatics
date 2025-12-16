#include "krykov_e_word_count/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <ranges>
#include <string>

#include "krykov_e_word_count/common/include/common.hpp"

namespace krykov_e_word_count {

KrykovEWordCountSEQ::KrykovEWordCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KrykovEWordCountSEQ::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool KrykovEWordCountSEQ::PreProcessingImpl() {
  auto &input = GetInput();
  input.erase(input.begin(), std::ranges::find_if(input, [](unsigned char ch) { return !std::isspace(ch); }));
  input.erase(
      std::ranges::find_if(std::ranges::reverse_view(input), [](unsigned char ch) { return !std::isspace(ch); }).base(),
      input.end());
  return true;
}

bool KrykovEWordCountSEQ::RunImpl() {
  const std::string &text = GetInput();

  bool in_word = false;
  size_t word_count = 0;

  for (char c : text) {
    if (std::isspace(static_cast<unsigned char>(c)) != 0) {
      if (in_word) {
        in_word = false;
      }
    } else {
      if (!in_word) {
        in_word = true;
        word_count++;
      }
    }
  }

  GetOutput() = static_cast<int>(word_count);
  return true;
}

bool KrykovEWordCountSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace krykov_e_word_count
