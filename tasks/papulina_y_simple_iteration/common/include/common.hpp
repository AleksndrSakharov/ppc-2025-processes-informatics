#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace papulina_y_simple_iteration {

using InType = std::tuple<size_t, std::vector<double>, std::vector<double>>;  // n, A, b
using OutType = std::vector<double>;
using TestType = std::string;  // название теста, (матрица A, матрица b,), ожидаемый результат
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace papulina_y_simple_iteration
