#include "romanova_v_min_by_matrix_rows_processes/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "romanova_v_min_by_matrix_rows_processes/common/include/common.hpp"

namespace romanova_v_min_by_matrix_rows_processes {

RomanovaVMinByMatrixRowsSEQ::RomanovaVMinByMatrixRowsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType(in.size());
}

bool RomanovaVMinByMatrixRowsSEQ::ValidationImpl() {
  return !GetInput().empty() && !GetInput()[0].empty();
}

bool RomanovaVMinByMatrixRowsSEQ::PreProcessingImpl() {
  in_data_ = GetInput();
  n_ = in_data_.size();
  m_ = in_data_[0].size();
  res_ = OutType(n_);
  return true;
}

bool RomanovaVMinByMatrixRowsSEQ::RunImpl() {
  for (size_t i = 0; i < n_; i++) {
    res_[i] = in_data_[i][0];
    for (size_t j = 1; j < m_; j++) {
      res_[i] = std::min(res_[i], in_data_[i][j]);
    }
  }
  return true;
}

bool RomanovaVMinByMatrixRowsSEQ::PostProcessingImpl() {
  GetOutput() = res_;
  return true;
}

}  // namespace romanova_v_min_by_matrix_rows_processes
