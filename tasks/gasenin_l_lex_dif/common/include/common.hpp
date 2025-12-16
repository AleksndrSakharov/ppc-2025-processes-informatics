#pragma once

#include <string>
#include <tuple>
#include <utility>

#include "task/include/task.hpp"

namespace gasenin_l_lex_dif {

using InType = std::pair<std::string, std::string>;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gasenin_l_lex_dif
