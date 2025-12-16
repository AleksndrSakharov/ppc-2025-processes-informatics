#pragma once

#include <cstddef>
#include <vector>

#include "task/include/task.hpp"

namespace kutuzov_i_elem_vec_average {

using InType = std::vector<double>;
using OutType = double;
using TestType = size_t;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kutuzov_i_elem_vec_average
