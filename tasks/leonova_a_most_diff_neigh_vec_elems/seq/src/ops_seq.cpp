#include "leonova_a_most_diff_neigh_vec_elems/seq/include/ops_seq.hpp"

#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

LeonovaAMostDiffNeighVecElemsSEQ::LeonovaAMostDiffNeighVecElemsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::tuple<int, int>(0, 0);
}

bool LeonovaAMostDiffNeighVecElemsSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool LeonovaAMostDiffNeighVecElemsSEQ::PreProcessingImpl() {
  return true;
}

bool LeonovaAMostDiffNeighVecElemsSEQ::RunImpl() {
  if (GetInput().size() == 1) {
    std::get<0>(GetOutput()) = GetInput()[0];
    std::get<1>(GetOutput()) = GetInput()[0];
    return true;
  }

  int64_t max_diff = -1;
  for (std::vector<int>::size_type index = 0; index < GetInput().size() - 1; index++) {
    std::tuple<int, int> curr_elems(GetInput()[index], GetInput()[index + 1]);

    int64_t curr_diff = std::llabs(static_cast<int64_t>(std::get<0>(curr_elems)) -
                                   static_cast<int64_t>(std::get<1>(curr_elems)));  // safe abs

    if (curr_diff > max_diff) {
      max_diff = curr_diff;
      GetOutput() = curr_elems;
    }
  }
  return true;
}

bool LeonovaAMostDiffNeighVecElemsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace leonova_a_most_diff_neigh_vec_elems
