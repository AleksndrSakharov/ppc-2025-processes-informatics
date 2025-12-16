#include "titaev_m_avg_el_vector/seq/include/ops_seq.hpp"

#include "titaev_m_avg_el_vector/common/include/common.hpp"

namespace titaev_m_avg_el_vector {

TitaevMElemVecsAvgSEQ::TitaevMElemVecsAvgSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TitaevMElemVecsAvgSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool TitaevMElemVecsAvgSEQ::PreProcessingImpl() {
  return true;
}

bool TitaevMElemVecsAvgSEQ::RunImpl() {
  const auto &vec = GetInput();
  double sum = 0.0;

  for (int v : vec) {
    sum += v;
  }

  GetOutput() = sum / static_cast<double>(vec.size());
  return true;
}

bool TitaevMElemVecsAvgSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_avg_el_vector
