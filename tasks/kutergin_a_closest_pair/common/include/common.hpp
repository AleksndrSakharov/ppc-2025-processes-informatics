#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kutergin_a_closest_pair {

using InType = std::vector<int>;
;
using OutType = int;
using TestType = std::tuple<std::tuple<InType, int>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kutergin_a_closest_pair
