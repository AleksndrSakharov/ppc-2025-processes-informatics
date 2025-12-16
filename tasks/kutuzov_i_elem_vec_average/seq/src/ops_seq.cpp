#include "kutuzov_i_elem_vec_average/seq/include/ops_seq.hpp"

#include <vector>

#include "kutuzov_i_elem_vec_average/common/include/common.hpp"

namespace kutuzov_i_elem_vec_average {

KutuzovIElemVecAverageSEQ::KutuzovIElemVecAverageSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool KutuzovIElemVecAverageSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool KutuzovIElemVecAverageSEQ::PreProcessingImpl() {
  return true;
}

bool KutuzovIElemVecAverageSEQ::RunImpl() {
  GetOutput() = 0.0;
  for (double x : GetInput()) {
    GetOutput() += x;
  }

  GetOutput() /= static_cast<double>(GetInput().size());

  return true;
}

bool KutuzovIElemVecAverageSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kutuzov_i_elem_vec_average
