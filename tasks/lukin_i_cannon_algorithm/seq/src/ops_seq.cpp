#include "lukin_i_cannon_algorithm/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "lukin_i_cannon_algorithm/common/include/common.hpp"

namespace lukin_i_cannon_algorithm {

LukinICannonAlgorithmSEQ::LukinICannonAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LukinICannonAlgorithmSEQ::ValidationImpl() {
  int rsize_a = static_cast<int>(std::get<0>(GetInput()).size());
  int rsize_b = static_cast<int>(std::get<1>(GetInput()).size());
  int size = std::get<2>(GetInput());
  return (rsize_a > 0) && (rsize_b > 0) && (sqrt(rsize_a) == size) && (rsize_a == rsize_b);
}

bool LukinICannonAlgorithmSEQ::PreProcessingImpl() {
  return true;
}

bool LukinICannonAlgorithmSEQ::RunImpl() {
  double *a = std::get<0>(GetInput()).data();
  double *b = std::get<1>(GetInput()).data();
  size_ = std::get<2>(GetInput());

  std::vector<double> c(static_cast<size_t>(size_ * size_), 0);
  double *cdata = c.data();

  for (int i = 0; i < size_; i++) {
    for (int k = 0; k < size_; k++) {
      double fixed = a[(i * size_) + k];
      for (int j = 0; j < size_; j++) {
        cdata[(i * size_) + j] += fixed * b[(k * size_) + j];
      }
    }
  }

  GetOutput() = std::move(c);
  return true;
}

bool LukinICannonAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_cannon_algorithm
