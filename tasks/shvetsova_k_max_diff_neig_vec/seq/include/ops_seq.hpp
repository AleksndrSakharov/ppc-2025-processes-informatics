#pragma once

#include <vector>

#include "shvetsova_k_max_diff_neig_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shvetsova_k_max_diff_neig_vec {

class ShvetsovaKMaxDiffNeigVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShvetsovaKMaxDiffNeigVecSEQ(const InType &in);

 private:
  std::vector<double> data_;
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shvetsova_k_max_diff_neig_vec
