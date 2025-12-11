#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace potashnik_m_char_freq {

using InType = std::tuple<std::string, char>;
using OutType = int;
using TestType = int;  // string size
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace potashnik_m_char_freq
