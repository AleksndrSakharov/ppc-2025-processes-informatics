#pragma once

#include <string>
#include <tuple>
#include <utility>

#include "task/include/task.hpp"

namespace egashin_k_lexicographical_check {

using InType = std::pair<std::string, std::string>;
using OutType = bool;

using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace egashin_k_lexicographical_check
