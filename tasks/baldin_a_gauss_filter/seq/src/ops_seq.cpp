#include "baldin_a_gauss_filter/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "baldin_a_gauss_filter/common/include/common.hpp"

namespace baldin_a_gauss_filter {

BaldinAGaussFilterSEQ::BaldinAGaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BaldinAGaussFilterSEQ::ValidationImpl() {
  const auto &im = GetInput();
  bool size_match = (im.pixels.size() == (static_cast<size_t>(im.width) * im.height * im.channels));
  return (im.width > 0 && im.height > 0 && im.channels > 0 && size_match);
}

bool BaldinAGaussFilterSEQ::PreProcessingImpl() {
  return true;
}

bool BaldinAGaussFilterSEQ::RunImpl() {
  ImageData &input = GetInput();
  ImageData res = input;
  int w = input.width;
  int h = input.height;
  int c = input.channels;

  constexpr std::array<int, 9> kKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};

  for (int row = 0; row < h; row++) {
    for (int col = 0; col < w; col++) {
      for (int ch = 0; ch < c; ch++) {
        int sum = 0;
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            int ny = std::clamp(row + dy, 0, h - 1);
            int nx = std::clamp(col + dx, 0, w - 1);

            int pixel_val = input.pixels[((ny * w + nx) * c) + ch];
            int kernel_val = kKernel.at((static_cast<size_t>(dy + 1) * 3) + (dx + 1));
            sum += pixel_val * kernel_val;
          }
        }
        res.pixels[((row * w + col) * c) + ch] = static_cast<uint8_t>(sum / 16);
      }
    }
  }

  GetOutput() = res;
  return true;
}

bool BaldinAGaussFilterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_gauss_filter
