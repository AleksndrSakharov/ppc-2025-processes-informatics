#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace guseva_a_matrix_sums {

constexpr double kEpsilon = 10e-12;
using InType = std::tuple<uint32_t, uint32_t, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace guseva_a_matrix_sums
