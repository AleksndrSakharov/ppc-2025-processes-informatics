#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace lifanov_k_adj_inv_count_restore {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace lifanov_k_adj_inv_count_restore
