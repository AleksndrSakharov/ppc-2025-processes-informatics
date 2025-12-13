#include "egashin_k_lexicographical_check/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>

#include "egashin_k_lexicographical_check/common/include/common.hpp"

namespace egashin_k_lexicographical_check {

TestTaskSEQ::TestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = false;
}

bool TestTaskSEQ::ValidationImpl() {
  return true;
}

bool TestTaskSEQ::PreProcessingImpl() {
  return true;
}

bool TestTaskSEQ::RunImpl() {
  const auto &s1 = GetInput().first;
  const auto &s2 = GetInput().second;

  const size_t len1 = s1.size();
  const size_t len2 = s2.size();

  size_t min_len = std::min(s1.size(), s2.size());

  for (size_t i = 0; i < min_len; ++i) {
    if (s1[i] < s2[i]) {
      GetOutput() = true;
      return true;
    }
    if (s1[i] > s2[i]) {
      GetOutput() = false;
      return true;
    }
  }

  GetOutput() = (len1 < len2);

  return true;
}

bool TestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace egashin_k_lexicographical_check
