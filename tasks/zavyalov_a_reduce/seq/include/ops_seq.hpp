#pragma once

#include "task/include/task.hpp"
#include "zavyalov_a_reduce/common/include/common.hpp"

namespace zavyalov_a_reduce {

class ZavyalovAReduceSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZavyalovAReduceSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zavyalov_a_reduce
