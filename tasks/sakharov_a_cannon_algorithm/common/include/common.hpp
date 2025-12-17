#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace sakharov_a_cannon_algorithm {

struct MatrixTaskInput {
  int size = 0;
  std::vector<double> a;
  std::vector<double> b;
};

using InType = MatrixTaskInput;
using OutType = std::vector<double>;
using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::size_t Offset(int size, int row, int col) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(size) + static_cast<std::size_t>(col);
}

inline bool HasValidShape(const InType &input) {
  if (input.size <= 0) {
    return false;
  }

  const std::size_t required = static_cast<std::size_t>(input.size) * static_cast<std::size_t>(input.size);
  return input.a.size() == required && input.b.size() == required;
}

inline int SelectBlockSize(int size) {
  constexpr int kTargetBytes = 96 * 1024;
  const int approx_block = static_cast<int>(std::sqrt(static_cast<double>(kTargetBytes) / sizeof(double)));
  return std::max(1, std::min(size, approx_block));
}

inline std::vector<double> BlockMultiply(const InType &input, int block_size) {
  const int n = input.size;
  std::vector<double> result(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);

  for (int ii = 0; ii < n; ii += block_size) {
    const int i_max = std::min(ii + block_size, n);
    for (int kk = 0; kk < n; kk += block_size) {
      const int k_max = std::min(kk + block_size, n);
      for (int jj = 0; jj < n; jj += block_size) {
        const int j_max = std::min(jj + block_size, n);
        for (int i = ii; i < i_max; ++i) {
          const std::size_t row_offset = Offset(n, i, 0);
          for (int k = kk; k < k_max; ++k) {
            const double a_val = input.a[row_offset + static_cast<std::size_t>(k)];
            const std::size_t b_row_offset = Offset(n, k, 0);
            for (int j = jj; j < j_max; ++j) {
              result[row_offset + static_cast<std::size_t>(j)] +=
                  a_val * input.b[b_row_offset + static_cast<std::size_t>(j)];
            }
          }
        }
      }
    }
  }

  return result;
}

}  // namespace sakharov_a_cannon_algorithm
