#pragma once

#include "task/include/task.hpp"
#include "titaev_m_avg_el_vector/common/include/common.hpp"

namespace titaev_m_avg_el_vector {

class TitaevMElemVecsAvgSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit TitaevMElemVecsAvgSEQ(const InType &in);

  ~TitaevMElemVecsAvgSEQ() override = default;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace titaev_m_avg_el_vector
