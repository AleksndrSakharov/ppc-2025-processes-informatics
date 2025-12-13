#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace lukin_i_torus_topology {

using InType = std::tuple<int, int, std::vector<int>>;           // откуда-куда + сообщение для валидации
using OutType = std::tuple<std::vector<int>, std::vector<int>>;  // путь + сообщение для валидации
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace lukin_i_torus_topology
