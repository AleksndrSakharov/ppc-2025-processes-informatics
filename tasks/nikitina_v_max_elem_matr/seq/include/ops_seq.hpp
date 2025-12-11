#pragma once

#include <vector>

#include "nikitina_v_max_elem_matr/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikitina_v_max_elem_matr {

class MaxElementMatrSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MaxElementMatrSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int rows_{};
  int cols_{};
  int max_val_{};
  std::vector<int> matrix_;
};

}  // namespace nikitina_v_max_elem_matr
