#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace sannikov_i_column_sum {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<std::vector<std::vector<int>>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sannikov_i_column_sum
