#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace rozenberg_a_matrix_column_sum {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace rozenberg_a_matrix_column_sum
