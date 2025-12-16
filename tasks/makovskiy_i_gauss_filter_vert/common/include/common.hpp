#pragma once

#include <algorithm>
#include <cstddef>
#include <task/include/task.hpp>
#include <tuple>
#include <vector>

namespace makovskiy_i_gauss_filter_vert {

using InType = std::tuple<std::vector<int>, int, int>;
using OutType = std::vector<int>;

using BaseTask = ppc::task::Task<InType, OutType>;

inline int GetPixel(const std::vector<int> &image, int x, int y, int width, int height) {
  int clamped_x = std::clamp(x, 0, width - 1);
  int clamped_y = std::clamp(y, 0, height - 1);
  return image[(static_cast<size_t>(clamped_y) * width) + clamped_x];
}

}  // namespace makovskiy_i_gauss_filter_vert
