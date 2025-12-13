#include "nikitin_a_vec_sign_rotation/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "nikitin_a_vec_sign_rotation/common/include/common.hpp"

namespace nikitin_a_vec_sign_rotation {

NikitinAVecSignRotationSEQ::NikitinAVecSignRotationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool NikitinAVecSignRotationSEQ::ValidationImpl() {
  return true;
}

bool NikitinAVecSignRotationSEQ::PreProcessingImpl() {
  return true;
}

bool NikitinAVecSignRotationSEQ::RunImpl() {
  int swaps = 0;
  std::vector<double> data = GetInput();

  // Проходим по вектору, ищем различия в знаках
  for (size_t i = 1; i < data.size(); i++) {
    if (IsSignChange(data[i - 1], data[i])) {
      swaps++;
    }
  }

  GetOutput() = swaps;
  return true;
}

bool NikitinAVecSignRotationSEQ::PostProcessingImpl() {
  return true;
}

bool NikitinAVecSignRotationSEQ::IsSignChange(double first_value, double second_value) {
  // Определяем знак каждого числа
  const bool first_negative = first_value < 0.0;
  const bool second_non_negative = second_value >= 0.0;
  const bool first_non_negative = first_value >= 0.0;
  const bool second_negative = second_value < 0.0;

  // Сравниваем знаки. true - знаки разные, false - знаки одинаковые
  return (first_negative && second_non_negative) || (first_non_negative && second_negative);
}

}  // namespace nikitin_a_vec_sign_rotation
