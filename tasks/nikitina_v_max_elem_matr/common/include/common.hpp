#pragma once

#include <vector>

#include "task/include/task.hpp"

namespace nikitina_v_max_elem_matr {

using InType = std::vector<int>;
using OutType = int;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nikitina_v_max_elem_matr
