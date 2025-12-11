#pragma once

#include "chaschin_v_max_for_each_row/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chaschin_v_max_for_each_row {

class ChaschinVMaxForEachRowSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ChaschinVMaxForEachRowSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace chaschin_v_max_for_each_row
