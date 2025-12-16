#include "makovskiy_i_gauss_filter_vert/seq/include/ops_seq.hpp"

#include <array>
#include <cstddef>
#include <task/include/task.hpp>
#include <tuple>
#include <vector>

#include "makovskiy_i_gauss_filter_vert/common/include/common.hpp"

namespace makovskiy_i_gauss_filter_vert {

GaussFilterSEQ::GaussFilterSEQ(const InType &in) {
  InType temp(in);
  this->GetInput().swap(temp);
  SetTypeOfTask(ppc::task::TypeOfTask::kSEQ);
}

bool GaussFilterSEQ::ValidationImpl() {
  const auto &[input, width, height] = GetInput();
  return !input.empty() && width > 0 && height > 0 && input.size() == static_cast<size_t>(width) * height;
}

bool GaussFilterSEQ::PreProcessingImpl() {
  const auto &[_, width, height] = GetInput();
  GetOutput().resize(static_cast<size_t>(width) * height);
  return true;
}

bool GaussFilterSEQ::RunImpl() {
  const auto &[input, width, height] = GetInput();
  auto &output = GetOutput();

  const std::array<int, 9> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  const int kernel_sum = 16;

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int sum = 0;
      for (int k_row = -1; k_row <= 1; ++k_row) {
        for (int k_col = -1; k_col <= 1; ++k_col) {
          sum += GetPixel(input, col + k_col, row + k_row, width, height) *
                 kernel.at((static_cast<size_t>(k_row + 1) * 3) + static_cast<size_t>(k_col + 1));
        }
      }
      output.at((static_cast<size_t>(row) * width) + col) = sum / kernel_sum;
    }
  }
  return true;
}

bool GaussFilterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace makovskiy_i_gauss_filter_vert
