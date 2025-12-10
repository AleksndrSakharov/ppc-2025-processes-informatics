#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace makovskiy_i_min_value_in_matrix_rows
