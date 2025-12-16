#pragma once

#include "smyshlaev_a_str_order_check/common/include/common.hpp"
#include "task/include/task.hpp"

namespace smyshlaev_a_str_order_check {

class SmyshlaevAStrOrderCheckMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SmyshlaevAStrOrderCheckMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool RunSequential(int min_len, int len1, int len2);
};

}  // namespace smyshlaev_a_str_order_check
