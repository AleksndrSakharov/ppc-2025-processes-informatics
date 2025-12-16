#include "chaschin_v_max_for_each_row/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "chaschin_v_max_for_each_row/common/include/common.hpp"

namespace chaschin_v_max_for_each_row {

ChaschinVMaxForEachRowSEQ::ChaschinVMaxForEachRowSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto in_copy = in;
  GetInput() = std::move(in_copy);
  this->GetOutput().clear();
}

bool ChaschinVMaxForEachRowSEQ::ValidationImpl() {
  const auto &in = GetInput();

  if (in.empty()) {
    return in.empty();
  }

  if (in[0].empty()) {
    return in[0].empty();
  }

  return true;
}

bool ChaschinVMaxForEachRowSEQ::PreProcessingImpl() {
  const auto &mat = GetInput();
  GetOutput().assign(mat.size(), 0.0F);
  return true;
}

bool ChaschinVMaxForEachRowSEQ::RunImpl() {
  const auto &mat = GetInput();
  auto &out = GetOutput();

  for (size_t i = 0; i < mat.size(); i++) {
    out[i] = *std::max_element(mat[i].begin(), mat[i].end());
  }

  return true;
}

bool ChaschinVMaxForEachRowSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace chaschin_v_max_for_each_row
