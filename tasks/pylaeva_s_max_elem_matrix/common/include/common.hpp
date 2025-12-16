#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace pylaeva_s_max_elem_matrix {

using InType = std::tuple<size_t, size_t, std::vector<int>>;
using OutType = int;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace pylaeva_s_max_elem_matrix
