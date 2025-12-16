#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace spichek_d_dot_product_of_vectors {

// Пара векторов как входные данные
using InType = std::pair<std::vector<int>, std::vector<int>>;
using OutType = int;
// Тестовые параметры: (пара_векторов, описание)
using TestType = std::tuple<std::pair<std::vector<int>, std::vector<int>>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace spichek_d_dot_product_of_vectors
