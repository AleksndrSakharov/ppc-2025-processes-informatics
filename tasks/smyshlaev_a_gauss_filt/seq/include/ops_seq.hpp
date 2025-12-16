#pragma once

#include "smyshlaev_a_gauss_filt/common/include/common.hpp"
#include "task/include/task.hpp"

namespace smyshlaev_a_gauss_filt {

class SmyshlaevAGaussFiltSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SmyshlaevAGaussFiltSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace smyshlaev_a_gauss_filt
