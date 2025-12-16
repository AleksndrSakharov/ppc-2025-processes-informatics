#include "rozenberg_a_matrix_column_sum/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "rozenberg_a_matrix_column_sum/common/include/common.hpp"

namespace rozenberg_a_matrix_column_sum {

RozenbergAMatrixColumnSumSEQ::RozenbergAMatrixColumnSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  InType empty;
  GetInput().swap(empty);

  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput().clear();
}

bool RozenbergAMatrixColumnSumSEQ::ValidationImpl() {
  return (!(GetInput().empty())) && (GetOutput().empty());
}

bool RozenbergAMatrixColumnSumSEQ::PreProcessingImpl() {
  GetOutput().resize(GetInput()[0].size());
  return GetOutput().size() == GetInput()[0].size();
}

bool RozenbergAMatrixColumnSumSEQ::RunImpl() {
  std::fill(GetOutput().begin(), GetOutput().end(), 0);

  for (auto &i : GetInput()) {
    for (size_t j = 0; j < i.size(); j++) {
      GetOutput()[j] += i[j];
    }
  }

  return true;
}

bool RozenbergAMatrixColumnSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace rozenberg_a_matrix_column_sum
