#pragma once

#include "kutuzov_i_elem_vec_average/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kutuzov_i_elem_vec_average {

class KutuzovIElemVecAverageSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KutuzovIElemVecAverageSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kutuzov_i_elem_vec_average
