#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace kondakov_v_min_val_in_matrix_str {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kondakov_v_min_val_in_matrix_str
