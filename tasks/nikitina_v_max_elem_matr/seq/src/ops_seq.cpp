#include "nikitina_v_max_elem_matr/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <vector>

#include "nikitina_v_max_elem_matr/common/include/common.hpp"

namespace nikitina_v_max_elem_matr {

MaxElementMatrSEQ::MaxElementMatrSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaxElementMatrSEQ::ValidationImpl() {
  const auto &in = GetInput();
  if (in.size() < 2) {
    return false;
  }
  rows_ = in[0];
  cols_ = in[1];
  return rows_ >= 0 && cols_ >= 0 && static_cast<size_t>(rows_) * cols_ == in.size() - 2;
}

bool MaxElementMatrSEQ::PreProcessingImpl() {
  const auto &in = GetInput();
  matrix_.clear();
  if (rows_ > 0 && cols_ > 0) {
    matrix_.reserve(static_cast<size_t>(rows_) * cols_);
    std::copy(in.begin() + 2, in.end(), std::back_inserter(matrix_));
  }
  max_val_ = std::numeric_limits<int>::min();
  return true;
}

bool MaxElementMatrSEQ::RunImpl() {
  if (matrix_.empty()) {
    max_val_ = std::numeric_limits<int>::min();
    return true;
  }
  max_val_ = matrix_[0];
  for (size_t i = 1; i < matrix_.size(); ++i) {
    max_val_ = std::max(max_val_, matrix_[i]);
  }
  return true;
}

bool MaxElementMatrSEQ::PostProcessingImpl() {
  GetOutput() = max_val_;
  return true;
}

}  // namespace nikitina_v_max_elem_matr
