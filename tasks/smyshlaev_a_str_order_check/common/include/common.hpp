#pragma once

#include <string>
#include <tuple>
#include <utility>

#include "task/include/task.hpp"

namespace smyshlaev_a_str_order_check {

using InType = std::pair<std::string, std::string>;
using OutType = int;
using TestType = std::tuple<std::string, std::string, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace smyshlaev_a_str_order_check
