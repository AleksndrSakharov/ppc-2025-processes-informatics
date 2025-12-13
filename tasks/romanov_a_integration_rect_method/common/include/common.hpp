#pragma once

#include <cmath>
#include <functional>
#include <tuple>

#include "task/include/task.hpp"

namespace romanov_a_integration_rect_method {

using InType = std::tuple<std::function<double(double)>, double, double, int>;
using OutType = double;
using TestType = std::tuple<std::function<double(double)>, double, double, int, double>;
;
using BaseTask = ppc::task::Task<InType, OutType>;

constexpr double kEps = 1e-4;

inline bool IsEqual(double a, double b) {
  return std::abs(a - b) <= kEps;
}

}  // namespace romanov_a_integration_rect_method
