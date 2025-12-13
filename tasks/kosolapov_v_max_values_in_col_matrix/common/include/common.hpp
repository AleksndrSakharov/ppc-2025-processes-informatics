#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kosolapov_v_max_values_in_col_matrix
