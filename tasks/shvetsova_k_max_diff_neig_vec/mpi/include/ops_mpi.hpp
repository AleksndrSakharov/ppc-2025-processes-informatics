#pragma once

#include <array>
#include <vector>

#include "shvetsova_k_max_diff_neig_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shvetsova_k_max_diff_neig_vec {

class ShvetsovaKMaxDiffNeigVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShvetsovaKMaxDiffNeigVecMPI(const InType &in);

 private:
  std::vector<double> data_;
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Доп функции
  static void CreateDistribution(int count_of_proc, int size_of_vector, std::vector<int> &count_elems,
                                 std::vector<int> &ind, int rank);
  static int WinnerRank(const std::vector<double> &all_diffs, int count_of_proc, int rank);
  static void CollectGlobalPair(int winner_rank, double local_a, double local_b, std::array<double, 2> &result_pair,
                                int count_of_proc);
  static void ProcessBoundaries(int count_of_proc, int rank, const std::vector<double> &part, int part_size,
                                double &local_diff, double &local_a, double &local_b);
};

}  // namespace shvetsova_k_max_diff_neig_vec
