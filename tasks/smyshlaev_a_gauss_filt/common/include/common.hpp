#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace smyshlaev_a_gauss_filt {
struct ImageType {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> data;
};

using InType = ImageType;
using OutType = ImageType;
using TestType = std::tuple<int, int, int, std::vector<uint8_t>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace smyshlaev_a_gauss_filt
