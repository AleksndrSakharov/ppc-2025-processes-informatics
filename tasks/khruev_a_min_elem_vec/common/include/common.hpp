#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace khruev_a_min_elem_vec {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<std::tuple<std::vector<int>, int>, std::string>;  // meta information
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace khruev_a_min_elem_vec
