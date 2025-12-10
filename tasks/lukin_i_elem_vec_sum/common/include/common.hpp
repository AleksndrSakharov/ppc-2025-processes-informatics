#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace lukin_i_elem_vec_sum {

using InType = std::vector<int>;
using OutType = int64_t;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace lukin_i_elem_vec_sum
