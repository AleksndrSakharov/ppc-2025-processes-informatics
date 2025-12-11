#pragma once

#include <vector>

#include "chaschin_v_max_for_each_row/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chaschin_v_max_for_each_row {

class ChaschinVMaxForEachRow : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChaschinVMaxForEachRow(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  struct RowRange {
    int start;
    int count;
  };

  static RowRange ComputeRange(int nrows, int rank, int size);
  static std::vector<std::vector<float>> DistributeRows(const std::vector<std::vector<float>> &mat, int rank, int size,
                                                        const RowRange &range);
  static std::vector<float> ComputeLocalMax(const std::vector<std::vector<float>> &local_mat);

  static void GatherResults(std::vector<float> &out, const std::vector<float> &local_out, int rank, int size,
                            const RowRange &range);
  static void SendRowsToWorkers(const std::vector<std::vector<float>> &mat, int size);
  static void ReceiveRowsFromRoot(std::vector<std::vector<float>> &local_mat);
};

}  // namespace chaschin_v_max_for_each_row
