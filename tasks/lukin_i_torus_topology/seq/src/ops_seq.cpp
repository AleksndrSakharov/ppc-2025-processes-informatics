#include "lukin_i_torus_topology/seq/include/ops_seq.hpp"

#include <tuple>
#include <vector>

#include "lukin_i_torus_topology/common/include/common.hpp"

namespace lukin_i_torus_topology {

LukinIThorTopologySEQ::LukinIThorTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(std::vector<int>{}, std::vector<int>{});
}

bool LukinIThorTopologySEQ::ValidationImpl() {
  return true;
}

bool LukinIThorTopologySEQ::PreProcessingImpl() {
  return true;
}

bool LukinIThorTopologySEQ::RunImpl() {
  const int size = 1e7;

  std::vector<int> tmp;
  tmp.reserve(size);

  for (int i = 0; i < size; i++) {
    tmp.push_back(i);
  }

  GetOutput() = std::make_tuple(dummy_route_, tmp);

  return true;
}

bool LukinIThorTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_torus_topology
