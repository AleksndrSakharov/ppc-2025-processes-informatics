#pragma once

#include <task/include/task.hpp>

#include "makovskiy_i_gauss_filter_vert/common/include/common.hpp"

namespace makovskiy_i_gauss_filter_vert {

class GaussFilterSEQ : public BaseTask {
 public:
  explicit GaussFilterSEQ(const InType &in);

  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

 protected:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace makovskiy_i_gauss_filter_vert
