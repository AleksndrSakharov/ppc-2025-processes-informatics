#pragma once

#include "ovsyannikov_n_num_mistm_in_two_str/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ovsyannikov_n_num_mistm_in_two_str {

class OvsyannikovNNumMistmInTwoStrSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit OvsyannikovNNumMistmInTwoStrSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace ovsyannikov_n_num_mistm_in_two_str
