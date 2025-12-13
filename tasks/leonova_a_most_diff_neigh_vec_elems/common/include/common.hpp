#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

using InType = std::vector<int>;
using OutType = std::tuple<int, int>;
using TestType = std::tuple<std::tuple<int, int>, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace leonova_a_most_diff_neigh_vec_elems
