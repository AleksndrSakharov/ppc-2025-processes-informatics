#include "zavyalov_a_scalar_product/seq/include/ops_seq.hpp"

#include <cstdlib>
#include <vector>

#include "zavyalov_a_scalar_product/common/include/common.hpp"

namespace zavyalov_a_scalar_product {

ZavyalovAScalarProductSEQ::ZavyalovAScalarProductSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool ZavyalovAScalarProductSEQ::ValidationImpl() {
  return (!std::get<0>(GetInput()).empty()) && (std::get<0>(GetInput()).size() == std::get<1>(GetInput()).size());
}

bool ZavyalovAScalarProductSEQ::PreProcessingImpl() {
  return true;
}

bool ZavyalovAScalarProductSEQ::RunImpl() {
  const auto &input = GetInput();
  const std::vector<double> &left = std::get<0>(input);
  const std::vector<double> &right = std::get<1>(input);

  double res = 0.0;
  for (size_t i = 0; i < left.size(); i++) {
    res += left[i] * right[i];
  }
  GetOutput() = res;
  return true;
}

bool ZavyalovAScalarProductSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_scalar_product
