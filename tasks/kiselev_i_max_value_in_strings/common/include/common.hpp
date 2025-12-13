#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kiselev_i_max_value_in_strings {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kiselev_i_max_value_in_strings
