#pragma once
#include <cstddef>
#include <vector>

#include "task/include/task.hpp"
#include "zenin_a_sum_values_by_columns_matrix/common/include/common.hpp"

namespace zenin_a_sum_values_by_columns_matrix {

class ZeninASumValuesByColumnsMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZeninASumValuesByColumnsMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void FillSendBuffer(const std::vector<double> &mat, std::vector<double> &sendbuf, size_t rows, size_t cols,
                             size_t base, size_t rest, int world_size);
};

}  // namespace zenin_a_sum_values_by_columns_matrix
