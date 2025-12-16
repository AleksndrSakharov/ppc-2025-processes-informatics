#include "shkrebko_m_count_char_freq/seq/include/ops_seq.hpp"

#include <algorithm>
#include <string>

#include "shkrebko_m_count_char_freq/common/include/common.hpp"

namespace shkrebko_m_count_char_freq {

ShkrebkoMCountCharFreqSEQ::ShkrebkoMCountCharFreqSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShkrebkoMCountCharFreqSEQ::ValidationImpl() {
  return (!std::get<0>(GetInput()).empty()) && (std::get<1>(GetInput()).length() == 1);
}

bool ShkrebkoMCountCharFreqSEQ::PreProcessingImpl() {
  return true;
}

bool ShkrebkoMCountCharFreqSEQ::RunImpl() {
  std::string str = std::get<0>(GetInput());
  std::string symbol_str = std::get<1>(GetInput());
  char symbol = symbol_str[0];
  auto result = std::count(str.begin(), str.end(), symbol);
  GetOutput() = static_cast<int>(result);
  return true;
}

bool ShkrebkoMCountCharFreqSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shkrebko_m_count_char_freq
