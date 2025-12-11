#include "sizov_d_string_mismatch_count/seq/include/ops_seq.hpp"

#include <cstddef>
#include <string>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"

namespace sizov_d_string_mismatch_count {

SizovDStringMismatchCountSEQ::SizovDStringMismatchCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SizovDStringMismatchCountSEQ::ValidationImpl() {
  const auto &[a, b] = GetInput();
  return !a.empty() && (a.size() == b.size());
}

bool SizovDStringMismatchCountSEQ::PreProcessingImpl() {
  const auto &[a, b] = GetInput();
  str_a_ = a;
  str_b_ = b;
  return true;
}

bool SizovDStringMismatchCountSEQ::RunImpl() {
  result_ = 0;
  const std::size_t n = str_a_.size();
  for (std::size_t i = 0; i < n; ++i) {
    if (str_a_[i] != str_b_[i]) {
      ++result_;
    }
  }
  return true;
}

bool SizovDStringMismatchCountSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace sizov_d_string_mismatch_count
