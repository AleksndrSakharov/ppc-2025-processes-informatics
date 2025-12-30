#include "pikhotskiy_r_multiplication_of_sparse_matrices/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pikhotskiy_r_multiplication_of_sparse_matrices {

SparseMatrixCRS DenseToCRS(const std::vector<double> &dense, int rows, int cols) {
  SparseMatrixCRS result(rows, cols);
  result.row_ptr.resize(rows + 1);
  result.row_ptr[0] = 0;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      double val = dense[(i * cols) + j];
      if (std::abs(val) > 1e-12) {
        result.values.push_back(val);
        result.col_indices.push_back(j);
      }
    }
    result.row_ptr[i + 1] = static_cast<int>(result.values.size());
  }
  return result;
}

std::vector<double> CRSToDense(const SparseMatrixCRS &sparse) {
  std::vector<double> dense(static_cast<size_t>(sparse.rows) * sparse.cols, 0.0);
  for (int i = 0; i < sparse.rows; ++i) {
    for (int k = sparse.row_ptr[i]; k < sparse.row_ptr[i + 1]; ++k) {
      dense[(i * sparse.cols) + sparse.col_indices[k]] = sparse.values[k];
    }
  }
  return dense;
}

SparseMatrixCRS TransposeCRS(const SparseMatrixCRS &matrix) {
  SparseMatrixCRS result(matrix.cols, matrix.rows);
  result.row_ptr.resize(matrix.cols + 1, 0);

  // Count elements in each column (will become rows in transposed)
  for (int col : matrix.col_indices) {
    result.row_ptr[col + 1]++;
  }

  // Cumulative sum to get row pointers
  for (int i = 1; i <= matrix.cols; ++i) {
    result.row_ptr[i] += result.row_ptr[i - 1];
  }

  result.values.resize(matrix.values.size());
  result.col_indices.resize(matrix.col_indices.size());

  std::vector<int> current_pos(result.row_ptr.begin(), result.row_ptr.end() - 1);

  for (int i = 0; i < matrix.rows; ++i) {
    for (int k = matrix.row_ptr[i]; k < matrix.row_ptr[i + 1]; ++k) {
      int col = matrix.col_indices[k];
      int pos = current_pos[col]++;
      result.values[pos] = matrix.values[k];
      result.col_indices[pos] = i;
    }
  }

  return result;
}

bool CompareSparseMatrices(const SparseMatrixCRS &a, const SparseMatrixCRS &b, double eps) {
  if (a.rows != b.rows || a.cols != b.cols) {
    return false;
  }
  auto dense_a = CRSToDense(a);
  auto dense_b = CRSToDense(b);
  for (size_t i = 0; i < dense_a.size(); ++i) {
    if (std::abs(dense_a[i] - dense_b[i]) > eps) {
      return false;
    }
  }
  return true;
}

SparseMatrixMultiplicationSEQ::SparseMatrixMultiplicationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SparseMatrixMultiplicationSEQ::ValidationImpl() {
  const auto &mat_a = std::get<0>(GetInput());
  const auto &mat_b = std::get<1>(GetInput());

  if (mat_a.cols != mat_b.rows) {
    return false;
  }

  if (mat_a.rows <= 0 || mat_a.cols <= 0 || mat_b.rows <= 0 || mat_b.cols <= 0) {
    return false;
  }

  if (mat_a.row_ptr.size() != static_cast<size_t>(mat_a.rows + 1)) {
    return false;
  }
  if (mat_b.row_ptr.size() != static_cast<size_t>(mat_b.rows + 1)) {
    return false;
  }

  return true;
}

bool SparseMatrixMultiplicationSEQ::PreProcessingImpl() {
  mat_a_ = std::get<0>(GetInput());
  mat_b_transposed_ = TransposeCRS(std::get<1>(GetInput()));
  return true;
}

bool SparseMatrixMultiplicationSEQ::RunImpl() {
  const auto &mat_b = std::get<1>(GetInput());
  SparseMatrixCRS result(mat_a_.rows, mat_b.cols);
  result.row_ptr.resize(static_cast<size_t>(mat_a_.rows) + 1);
  if (!result.row_ptr.empty()) {
    result.row_ptr[0] = 0;
  }

  for (int i = 0; i < mat_a_.rows; ++i) {
    for (int j = 0; j < mat_b.cols; ++j) {
      double sum = 0.0;

      int a_start = mat_a_.row_ptr[i];
      int a_end = mat_a_.row_ptr[i + 1];

      int bt_start = mat_b_transposed_.row_ptr[j];
      int bt_end = mat_b_transposed_.row_ptr[j + 1];

      int a_idx = a_start;
      int bt_idx = bt_start;

      while (a_idx < a_end && bt_idx < bt_end) {
        int a_col = mat_a_.col_indices[a_idx];
        int bt_col = mat_b_transposed_.col_indices[bt_idx];

        if (a_col == bt_col) {
          sum += mat_a_.values[a_idx] * mat_b_transposed_.values[bt_idx];
          ++a_idx;
          ++bt_idx;
        } else if (a_col < bt_col) {
          ++a_idx;
        } else {
          ++bt_idx;
        }
      }

      if (std::abs(sum) > 1e-12) {
        result.values.push_back(sum);
        result.col_indices.push_back(j);
      }
    }
    result.row_ptr[i + 1] = static_cast<int>(result.values.size());
  }

  GetOutput() = result;
  return true;
}

bool SparseMatrixMultiplicationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace pikhotskiy_r_multiplication_of_sparse_matrices
