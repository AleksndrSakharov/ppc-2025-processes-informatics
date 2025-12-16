#include "vector_scalar_product/seq/include/ops_seq.hpp"

#include <numeric>

#include "vector_scalar_product/common/include/common.hpp"

namespace vector_scalar_product {

VectorScalarProductSeq::VectorScalarProductSeq(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool VectorScalarProductSeq::ValidationImpl() {
  const auto &lhs = GetInput().lhs;
  const auto &rhs = GetInput().rhs;
  return !lhs.empty() && lhs.size() == rhs.size();
}

bool VectorScalarProductSeq::PreProcessingImpl() {
  partial_sum_ = 0.0;
  return true;
}

bool VectorScalarProductSeq::RunImpl() {
  const auto &lhs = GetInput().lhs;
  const auto &rhs = GetInput().rhs;
  partial_sum_ = std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), 0.0);
  return true;
}

bool VectorScalarProductSeq::PostProcessingImpl() {
  GetOutput() = partial_sum_;
  return true;
}

}  // namespace vector_scalar_product
