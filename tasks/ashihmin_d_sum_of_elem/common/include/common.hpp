#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace ashihmin_d_sum_of_elem {

using InType = std::vector<int>;
using OutType = int64_t;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace ashihmin_d_sum_of_elem
