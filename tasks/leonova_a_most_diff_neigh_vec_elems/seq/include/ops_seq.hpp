#pragma once

#include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
#include "task/include/task.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

class LeonovaAMostDiffNeighVecElemsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LeonovaAMostDiffNeighVecElemsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace leonova_a_most_diff_neigh_vec_elems
