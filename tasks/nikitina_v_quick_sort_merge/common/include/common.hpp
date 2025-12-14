#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace nikitina_v_quick_sort_merge {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::pair<int, int> Partition(std::vector<int> &vec, int left, int right) {
  int i = left;
  int j = right;
  int pivot = vec[(left + right) / 2];

  while (i <= j) {
    while (vec[i] < pivot) {
      i++;
    }
    while (vec[j] > pivot) {
      j--;
    }
    if (i <= j) {
      std::swap(vec[i], vec[j]);
      i++;
      j--;
    }
  }
  return {i, j};
}

inline void QuickSortImpl(std::vector<int> &vec, int left, int right) {
  if (left >= right) {
    return;
  }

  std::vector<std::pair<int, int>> stack;
  stack.reserve(static_cast<size_t>(right - left) + 1);
  stack.emplace_back(left, right);

  while (!stack.empty()) {
    auto [l, r] = stack.back();
    stack.pop_back();

    auto [i, j] = Partition(vec, l, r);

    if (l < j) {
      stack.emplace_back(l, j);
    }
    if (i < r) {
      stack.emplace_back(i, r);
    }
  }
}

}  // namespace nikitina_v_quick_sort_merge
