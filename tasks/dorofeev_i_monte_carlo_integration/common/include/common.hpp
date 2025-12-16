#pragma once

#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace dorofeev_i_monte_carlo_integration_processes {

struct InputData {
  std::function<double(const std::vector<double> &)> func;  // f(x)
  std::vector<double> a;                                    // lower bounds
  std::vector<double> b;                                    // upper bounds
  int samples = 0;                                          // number of samples
};

using InType = InputData;
using OutType = double;

using TestType = std::tuple<int, std::string>;  // for testing purposes
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace dorofeev_i_monte_carlo_integration_processes
