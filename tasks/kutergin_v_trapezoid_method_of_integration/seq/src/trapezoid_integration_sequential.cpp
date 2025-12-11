#include "../include/trapezoid_integration_sequential.hpp"

#include "../../common/include/common.hpp"

namespace kutergin_v_trapezoid_seq {

double Func(double x)  // интегрируемая функция для примера
{
  return x * x;
}

TrapezoidIntegrationSequential::TrapezoidIntegrationSequential(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());  // установка типа задачи
  GetInput() = in;                       // сохранение входных данных
  GetOutput() = 0.0;                     // инициализация выходных данных
}

bool TrapezoidIntegrationSequential::ValidationImpl() {
  return (GetInput().b >= GetInput().a) &&
         (GetInput().n > 0);  // проверка b >= a (границ интегрирования) и n > 0 (число разбиений)
}

bool TrapezoidIntegrationSequential::PreProcessingImpl() {
  return true;
}

bool TrapezoidIntegrationSequential::RunImpl() {
  double a = GetInput().a;
  double b = GetInput().b;
  int n = GetInput().n;

  double h = (b - a) / n;
  double integral_res = (Func(a) + Func(b)) / 2.0;

  for (int i = 1; i < n; ++i) {
    integral_res += Func(a + (i * h));
  }

  GetOutput() = integral_res * h;

  return true;
}

bool TrapezoidIntegrationSequential::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_v_trapezoid_seq
