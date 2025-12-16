#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace egorova_l_find_max_val_col_matrix {
// задала подходящие под мою задачу типы данных
// на вход принимаю матрицу заданную парой векторов
// на выход передается вектор максимальных значений столбцов матрицы
using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace egorova_l_find_max_val_col_matrix
