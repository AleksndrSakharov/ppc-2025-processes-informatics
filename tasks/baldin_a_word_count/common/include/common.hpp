#pragma once

#include <cstddef>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace baldin_a_word_count {

using InType = std::string;
using OutType = size_t;
using TestType = std::tuple<std::string, size_t>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace baldin_a_word_count
