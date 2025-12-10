#include "votincev_d_alternating_values/seq/include/ops_seq.hpp"

#include <cstddef>  // для size_t
#include <vector>

#include "votincev_d_alternating_values/common/include/common.hpp"

namespace votincev_d_alternating_values {

VotincevDAlternatingValuesSEQ::VotincevDAlternatingValuesSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool VotincevDAlternatingValuesSEQ::ValidationImpl() {
  return true;
}

bool VotincevDAlternatingValuesSEQ::PreProcessingImpl() {
  return true;
}

bool VotincevDAlternatingValuesSEQ::RunImpl() {
  int all_swaps = 0;
  std::vector<double> vect_data = GetInput();
  for (size_t i = 1; i < vect_data.size(); i++) {
    if (IsSignChange(vect_data[i - 1], vect_data[i])) {
      all_swaps++;
    }
  }
  GetOutput() = all_swaps;
  return true;
}

// вспомогательная, if с такими условиями выглядит страшно
bool VotincevDAlternatingValuesSEQ::IsSignChange(const double &a, const double &b) {
  return ((a < 0 && b >= 0) || (a >= 0 && b < 0));
}

bool VotincevDAlternatingValuesSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_alternating_values
