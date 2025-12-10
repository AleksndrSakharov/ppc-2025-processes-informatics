#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace romanova_v_min_by_matrix_rows_processes {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace romanova_v_min_by_matrix_rows_processes
