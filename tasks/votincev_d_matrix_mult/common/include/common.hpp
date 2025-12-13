#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace votincev_d_matrix_mult {
//                         m   n   k      A (m x k)          B (k x n)
using InType = std::tuple<int, int, int, std::vector<double>, std::vector<double>>;
using OutType = std::vector<double>;  // R (m x n)   результирующая матрица
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace votincev_d_matrix_mult
