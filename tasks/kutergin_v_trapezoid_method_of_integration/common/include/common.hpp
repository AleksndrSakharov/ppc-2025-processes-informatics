#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace kutergin_v_trapezoid_seq {

struct InputData {
  double a;
  double b;
  int n;
};

using InType = InputData;
using OutType = double;
using BaseTask = ppc::task::Task<InType, OutType>;
using TestType = std::tuple<InputData, double, std::string>;  // Тип для тестовых параметров {входные данные, ожидаемый
                                                              // результат, имя_теста}

}  // namespace kutergin_v_trapezoid_seq
