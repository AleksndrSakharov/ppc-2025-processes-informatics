#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "kondakov_v_min_val_in_matrix_str/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kondakov_v_min_val_in_matrix_str {

class KondakovVMinValMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KondakovVMinValMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::pair<size_t, size_t> BroadcastMatrixDimensions();
  std::vector<int> ScatterMatrixData(size_t total_rows, size_t cols, int n, int rank);
  static std::vector<int> ComputeLocalMinima(const std::vector<int> &local_flat, size_t cols, int local_row_count);
  static void PrepareSendInfo(int rank, size_t total_rows, size_t cols, int n, std::vector<int> &send_counts,
                              std::vector<int> &send_displs);
  static std::vector<int> GatherMinima(const std::vector<int> &local_minima, size_t total_rows, int n, int rank);
};
}  // namespace kondakov_v_min_val_in_matrix_str
