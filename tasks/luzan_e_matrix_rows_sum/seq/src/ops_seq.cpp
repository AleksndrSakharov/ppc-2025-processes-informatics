#include "luzan_e_matrix_rows_sum/seq/include/ops_seq.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

#include "luzan_e_matrix_rows_sum/common/include/common.hpp"

namespace luzan_e_matrix_rows_sum {

LuzanEMatrixRowsSumSEQ::LuzanEMatrixRowsSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LuzanEMatrixRowsSumSEQ::ValidationImpl() {
  int height = std::get<1>(GetInput());
  int width = std::get<2>(GetInput());

  return std::get<0>(GetInput()).size() == static_cast<size_t>(height) * static_cast<size_t>(width) && height > 0 &&
         width > 0;
}

bool LuzanEMatrixRowsSumSEQ::PreProcessingImpl() {
  int height = std::get<1>(GetInput());
  GetOutput().resize(height);
  for (int row = 0; row < height; row++) {
    GetOutput()[row] = 0;
  }
  return true;
}

bool LuzanEMatrixRowsSumSEQ::RunImpl() {
  int height = std::get<1>(GetInput());
  int width = std::get<2>(GetInput());
  const std::tuple_element_t<0, InType> &mat = std::get<0>(GetInput());

  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      GetOutput()[row] += mat[(width * row) + col];
    }
  }
  return true;
}

bool LuzanEMatrixRowsSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace luzan_e_matrix_rows_sum
