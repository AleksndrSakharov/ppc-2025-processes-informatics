#pragma once

#include <vector>

#include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
#include "task/include/task.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

class LeonovaAMostDiffNeighVecElemsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LeonovaAMostDiffNeighVecElemsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ProcessWithMultipleProcesses(int rank, int size, int total_size, const std::vector<int> &input_vec);
  static void ProcessLocalData(int rank, int actual_processes, int total_size, const std::vector<int> &input_vec,
                               int &local_max_diff, int &local_first, int &local_second, int size);
  static void ReceiveLocalData(int rank, int actual_processes, const std::vector<int> &input_vec,
                               const std::vector<int> &sizes, const std::vector<int> &offsets,
                               std::vector<int> &local_data, int size);
  static void FindLocalMaxDiff(const std::vector<int> &local_data, int &local_max_diff, int &local_first,
                               int &local_second);
  void GatherAndProcessResults(int rank, int actual_processes, int local_max_diff, int local_first, int local_second,
                               int size);
  void BroadcastResult(int rank);
};

}  // namespace leonova_a_most_diff_neigh_vec_elems
