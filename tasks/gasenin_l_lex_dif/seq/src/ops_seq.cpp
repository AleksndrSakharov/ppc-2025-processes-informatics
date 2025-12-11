#include "gasenin_l_lex_dif/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>

#include "gasenin_l_lex_dif/common/include/common.hpp"

namespace gasenin_l_lex_dif {

GaseninLLexDifSEQ::GaseninLLexDifSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GaseninLLexDifSEQ::ValidationImpl() {
  const auto &[str1, str2] = GetInput();
  return str1.length() <= 100000000 && str2.length() <= 100000000;
}

bool GaseninLLexDifSEQ::PreProcessingImpl() {
  return true;
}

bool GaseninLLexDifSEQ::RunImpl() {
  const auto &[str1, str2] = GetInput();

  size_t min_len = std::min(str1.length(), str2.length());

  for (size_t i = 0; i < min_len; ++i) {
    if (str1[i] != str2[i]) {
      GetOutput() = (str1[i] < str2[i]) ? -1 : 1;
      return true;
    }
  }

  if (str1.length() < str2.length()) {
    GetOutput() = -1;
  } else if (str1.length() > str2.length()) {
    GetOutput() = 1;
  } else {
    GetOutput() = 0;
  }

  return true;
}

bool GaseninLLexDifSEQ::PostProcessingImpl() {
  return true;
}
}  // namespace gasenin_l_lex_dif
