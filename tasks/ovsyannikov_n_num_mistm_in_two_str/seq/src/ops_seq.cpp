#include "ovsyannikov_n_num_mistm_in_two_str/seq/include/ops_seq.hpp"

#include <cstddef>

// Clang-Tidy требует явного подключения файла, где определен InType
#include "ovsyannikov_n_num_mistm_in_two_str/common/include/common.hpp"

namespace ovsyannikov_n_num_mistm_in_two_str {

OvsyannikovNNumMistmInTwoStrSEQ::OvsyannikovNNumMistmInTwoStrSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool OvsyannikovNNumMistmInTwoStrSEQ::ValidationImpl() {
  return GetInput().first.size() == GetInput().second.size();
}

bool OvsyannikovNNumMistmInTwoStrSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool OvsyannikovNNumMistmInTwoStrSEQ::RunImpl() {
  const auto &seq_one = GetInput().first;
  const auto &seq_two = GetInput().second;
  int diff_cnt = 0;

  size_t length = seq_one.size();
  for (size_t i = 0; i < length; ++i) {
    if (seq_one[i] != seq_two[i]) {
      diff_cnt++;
    }
  }

  GetOutput() = diff_cnt;
  return true;
}

bool OvsyannikovNNumMistmInTwoStrSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ovsyannikov_n_num_mistm_in_two_str
