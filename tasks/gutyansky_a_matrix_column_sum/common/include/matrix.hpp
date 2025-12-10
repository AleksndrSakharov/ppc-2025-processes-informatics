#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

namespace gutyansky_a_matrix_column_sum {

struct Matrix {
  size_t rows;
  size_t cols;
  std::vector<int32_t> data;

  [[nodiscard]] bool IsValid() const {
    return rows > 0 && cols > 0 && data.size() == rows * cols;
  }

  friend bool operator==(const Matrix &v_left, const Matrix &v_right) {
    return std::tie(v_left.rows, v_left.cols, v_left.data) == std::tie(v_right.rows, v_right.cols, v_right.data);
  }
};

}  // namespace gutyansky_a_matrix_column_sum
