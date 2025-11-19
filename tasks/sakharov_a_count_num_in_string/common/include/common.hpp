#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace sakharov_a_count_num_in_string {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sakharov_a_count_num_in_string
