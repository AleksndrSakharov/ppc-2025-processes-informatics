#pragma once

#include <vector>

#include "lukin_i_torus_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lukin_i_torus_topology {

class LukinIThorTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LukinIThorTopologySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> dummy_route_ = {2, 0, 2, 6};
};
}  // namespace lukin_i_torus_topology
