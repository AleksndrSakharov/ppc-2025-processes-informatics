#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace lukin_i_cannon_algorithm {

using InType = std::tuple<std::vector<double>, std::vector<double>, int>;
using OutType = std::vector<double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace lukin_i_cannon_algorithm
