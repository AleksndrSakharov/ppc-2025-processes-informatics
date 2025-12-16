#pragma once

#include "pylaeva_s_max_elem_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pylaeva_s_max_elem_matrix {

class PylaevaSMaxElemMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PylaevaSMaxElemMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace pylaeva_s_max_elem_matrix
