#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace sakharov_a_num_of_letters {

using InType = std::tuple<int, std::string>;
using OutType = int;
using TestType = std::tuple<std::tuple<int, std::string>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sakharov_a_num_of_letters
