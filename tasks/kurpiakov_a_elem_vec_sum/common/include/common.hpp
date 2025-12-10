#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kurpiakov_a_elem_vec_sum {
using InType = std::tuple<int, std::vector<int>>;
using OutType = int64_t;
using TestType = std::tuple<InType, std::string, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kurpiakov_a_elem_vec_sum
