#include "spichek_d_dot_product_of_vectors/seq/include/ops_seq.hpp"

#include <cstddef>

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"

namespace spichek_d_dot_product_of_vectors {

SpichekDDotProductOfVectorsSEQ::SpichekDDotProductOfVectorsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDDotProductOfVectorsSEQ::ValidationImpl() {
  const auto &[vector1, vector2] = GetInput();
  return vector1.size() == vector2.size();
}

bool SpichekDDotProductOfVectorsSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDDotProductOfVectorsSEQ::RunImpl() {
  const auto &[vector1, vector2] = GetInput();

  if (vector1.empty() && vector2.empty()) {
    GetOutput() = 0;
    return true;
  }

  int dot_product = 0;
  for (size_t i = 0; i < vector1.size(); ++i) {
    dot_product += vector1[i] * vector2[i];
  }

  GetOutput() = dot_product;
  return true;
}

bool SpichekDDotProductOfVectorsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
