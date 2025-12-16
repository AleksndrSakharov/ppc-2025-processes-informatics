#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include "task/include/task.hpp"

namespace maslova_u_char_frequency_count {

using InType = std::pair<std::string, char>;
using OutType = size_t;
using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace maslova_u_char_frequency_count
