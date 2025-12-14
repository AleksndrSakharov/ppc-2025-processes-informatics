#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace baldin_a_gauss_filter {

struct ImageData {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> pixels;

  bool operator==(const ImageData &other) const {
    return width == other.width && height == other.height && channels == other.channels && pixels == other.pixels;
  }
};

using InType = ImageData;
using OutType = ImageData;
using TestType = std::tuple<int, int, int>;  // width, height, channels
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace baldin_a_gauss_filter
