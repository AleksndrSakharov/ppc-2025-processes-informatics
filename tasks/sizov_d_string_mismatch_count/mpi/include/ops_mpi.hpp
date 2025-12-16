#pragma once

#include <string>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sizov_d_string_mismatch_count {

class SizovDStringMismatchCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit SizovDStringMismatchCountMPI(const InType &in);

 private:
  std::string str_a_;
  std::string str_b_;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sizov_d_string_mismatch_count
