#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "votincev_d_alternating_values/common/include/common.hpp"

namespace votincev_d_alternating_values {

class VotincevDAlternatingValuesMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VotincevDAlternatingValuesMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // == мои дополнительные функции ==
  static int ProcessMaster(int process_n, const std::vector<double> &vect_data);
  static void ProcessWorker();
  static bool IsSignChange(const double &a, const double &b);
  // ================================
};

}  // namespace votincev_d_alternating_values
