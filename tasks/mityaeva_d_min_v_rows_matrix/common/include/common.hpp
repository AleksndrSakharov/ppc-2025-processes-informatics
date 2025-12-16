#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace mityaeva_d_min_v_rows_matrix {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace mityaeva_d_min_v_rows_matrix
