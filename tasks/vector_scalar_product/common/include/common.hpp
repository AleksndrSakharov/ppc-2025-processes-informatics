#pragma once

#include <vector>

#include "task/include/task.hpp"

namespace vector_scalar_product {

struct DotProductInput {
  std::vector<double> lhs;
  std::vector<double> rhs;
};

using InType = DotProductInput;
using OutType = double;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace vector_scalar_product
