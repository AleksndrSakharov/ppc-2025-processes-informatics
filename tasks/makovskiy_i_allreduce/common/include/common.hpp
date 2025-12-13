#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace makovskiy_i_allreduce {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace makovskiy_i_allreduce
