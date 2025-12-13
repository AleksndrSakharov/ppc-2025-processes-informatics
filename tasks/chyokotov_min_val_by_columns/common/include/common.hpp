#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace chyokotov_min_val_by_columns {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<std::vector<std::vector<int>>, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace chyokotov_min_val_by_columns
