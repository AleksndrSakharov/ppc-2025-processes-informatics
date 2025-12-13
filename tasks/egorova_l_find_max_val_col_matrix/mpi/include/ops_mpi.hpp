#pragma once

#include <vector>

#include "egorova_l_find_max_val_col_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egorova_l_find_max_val_col_matrix {

class EgorovaLFindMaxValColMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit EgorovaLFindMaxValColMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Объявления вспомогательных методов
  bool RunMPIAlgorithm();
  std::vector<int> GetLocalMatrixPart(int rank, int size, int rows, int cols, int start_col, int local_cols_count);
  void SendMatrixPartsToOtherRanks(int size, int rows, int cols);
  std::vector<int> PrepareMatrixPartForRank(int dest_rank, int size, int rows, int cols);
  static std::vector<int> CalculateLocalMaxima(const std::vector<int> &local_matrix_part, int rows,
                                               int local_cols_count);
  static std::vector<int> GatherResults(const std::vector<int> &local_max, int size, int cols);
};

}  // namespace egorova_l_find_max_val_col_matrix
