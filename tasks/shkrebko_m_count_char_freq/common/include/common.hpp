#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace shkrebko_m_count_char_freq {

using InType = std::tuple<std::string, std::string>;
using OutType = int;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shkrebko_m_count_char_freq
