#pragma once

#include <string>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sizov_d_string_mismatch_count {

class SizovDStringMismatchCountSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit SizovDStringMismatchCountSEQ(const InType &in);

 private:
  std::string str_a_;
  std::string str_b_;
  int result_ = 0;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sizov_d_string_mismatch_count
