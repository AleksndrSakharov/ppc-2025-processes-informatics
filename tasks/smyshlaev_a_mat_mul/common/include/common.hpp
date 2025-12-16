#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace smyshlaev_a_mat_mul {

using InType = std::tuple<int, std::vector<double>, int, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::vector<double>, int, std::vector<double>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace smyshlaev_a_mat_mul
