#pragma once

#include <vector>

#include "khruev_a_min_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace khruev_a_min_elem_vec {

class KhruevAMinElemVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KhruevAMinElemVecSEQ(const InType &in);

 private:
  std::vector<int> data_;
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace khruev_a_min_elem_vec
