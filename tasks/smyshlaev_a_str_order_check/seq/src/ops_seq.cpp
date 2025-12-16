#include "smyshlaev_a_str_order_check/seq/include/ops_seq.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include "smyshlaev_a_str_order_check/common/include/common.hpp"

namespace smyshlaev_a_str_order_check {

SmyshlaevAStrOrderCheckSEQ::SmyshlaevAStrOrderCheckSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SmyshlaevAStrOrderCheckSEQ::ValidationImpl() {
  return true;
}

bool SmyshlaevAStrOrderCheckSEQ::PreProcessingImpl() {
  return true;
}

bool SmyshlaevAStrOrderCheckSEQ::RunImpl() {
  const auto &input_data = GetInput();

  const std::string &str1 = input_data.first;
  const std::string &str2 = input_data.second;

  int min_len = static_cast<int>(std::min(str1.length(), str2.length()));

  for (int i = 0; i < min_len; ++i) {
    if (str1[i] < str2[i]) {
      GetOutput() = -1;
      return true;
    }
    if (str1[i] > str2[i]) {
      GetOutput() = 1;
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

bool SmyshlaevAStrOrderCheckSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_str_order_check
