#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace telnov_transfer_one_all {

using TestType = std::tuple<int, std::string>;

using InTypeInt = std::vector<int>;
using OutTypeInt = InTypeInt;

using InTypeFloat = std::vector<float>;
using OutTypeFloat = InTypeFloat;

using InTypeDouble = std::vector<double>;
using OutTypeDouble = InTypeDouble;

using InType = InTypeInt;
using OutType = OutTypeInt;

template <typename T>
using BaseTask = ppc::task::Task<std::vector<T>, std::vector<T>>;

}  // namespace telnov_transfer_one_all
