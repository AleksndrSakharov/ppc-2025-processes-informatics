#pragma once

#include "smyshlaev_a_str_order_check/common/include/common.hpp"
#include "task/include/task.hpp"

namespace smyshlaev_a_str_order_check {

class SmyshlaevAStrOrderCheckSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SmyshlaevAStrOrderCheckSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace smyshlaev_a_str_order_check
