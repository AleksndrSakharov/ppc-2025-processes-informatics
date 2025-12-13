#pragma once

#include "smyshlaev_a_mat_mul/common/include/common.hpp"
#include "task/include/task.hpp"

namespace smyshlaev_a_mat_mul {

class SmyshlaevAMatMulSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SmyshlaevAMatMulSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace smyshlaev_a_mat_mul
