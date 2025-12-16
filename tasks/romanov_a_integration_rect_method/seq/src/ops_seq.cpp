#include "romanov_a_integration_rect_method/seq/include/ops_seq.hpp"

#include <cmath>

#include "romanov_a_integration_rect_method/common/include/common.hpp"

namespace romanov_a_integration_rect_method {

RomanovAIntegrationRectMethodSEQ::RomanovAIntegrationRectMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool RomanovAIntegrationRectMethodSEQ::ValidationImpl() {
  if (!IsEqual(GetOutput(), 0.0)) {
    return false;
  }
  if (std::get<3>(GetInput()) <= 0) {
    return false;
  }
  if (std::get<1>(GetInput()) >= std::get<2>(GetInput())) {
    return false;
  }
  return true;
}

bool RomanovAIntegrationRectMethodSEQ::PreProcessingImpl() {
  return true;
}

bool RomanovAIntegrationRectMethodSEQ::RunImpl() {
  const auto &[f, a, b, n] = GetInput();

  double delta_x = (b - a) / static_cast<double>(n);
  double mid = a + (delta_x / 2.0);

  double result = 0.0;

  for (int i = 0; i < n; ++i) {
    result += f(mid) * delta_x;
    mid += delta_x;
  }

  GetOutput() = result;

  return true;
}

bool RomanovAIntegrationRectMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_integration_rect_method
