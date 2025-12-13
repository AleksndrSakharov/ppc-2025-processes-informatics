#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace telnov_counting_the_frequency {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

struct GlobalData {
  inline static std::string g_data_string{};
};

}  // namespace telnov_counting_the_frequency
