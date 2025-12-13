#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include "task/include/task.hpp"

namespace ovsyannikov_n_num_mistm_in_two_str {
using InType = std::pair<std::string, std::string>;  // Входные данные: пара строк
using OutType = size_t;                              // Выходные данные: количество несовпадений
using BaseTask = ppc::task::Task<InType, OutType>;
using TestType = std::tuple<InType, OutType, std::string>;
}  // namespace ovsyannikov_n_num_mistm_in_two_str
