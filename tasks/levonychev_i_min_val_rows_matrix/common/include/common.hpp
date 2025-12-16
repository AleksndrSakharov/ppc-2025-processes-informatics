#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace levonychev_i_min_val_rows_matrix {

using InType = std::tuple<std::vector<int>, int, int>;
using OutType = std::vector<int>;
using TestType = std::tuple<bool, int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace levonychev_i_min_val_rows_matrix
