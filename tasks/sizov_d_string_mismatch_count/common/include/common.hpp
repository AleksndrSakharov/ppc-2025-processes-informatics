#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace sizov_d_string_mismatch_count {

using InType = std::tuple<std::string, std::string>;
using OutType = int;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sizov_d_string_mismatch_count
