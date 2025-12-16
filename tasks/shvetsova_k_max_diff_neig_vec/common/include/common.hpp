#pragma once

#include <string>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace shvetsova_k_max_diff_neig_vec {

using InType = std::vector<double>;
using OutType = std::pair<double, double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shvetsova_k_max_diff_neig_vec
