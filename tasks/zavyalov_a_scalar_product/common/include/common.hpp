#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace zavyalov_a_scalar_product {

using InType = std::tuple<std::vector<double>, std::vector<double>>;
using OutType = double;
using TestType = unsigned int;  // size of vectors
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zavyalov_a_scalar_product
