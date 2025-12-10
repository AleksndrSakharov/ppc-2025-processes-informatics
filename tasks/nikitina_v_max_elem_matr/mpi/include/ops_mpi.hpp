#pragma once

#include <vector>

#include "nikitina_v_max_elem_matr/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikitina_v_max_elem_matr {

class MaxElementMatrMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MaxElementMatrMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void CalculateScatterParams(int total_elements, int world_size, std::vector<int> &sendcounts,
                                     std::vector<int> &displs);
  static int FindLocalMax(const std::vector<int> &data);

  int rows_{};
  int cols_{};
  int global_max_{};
  std::vector<int> matrix_;
};

}  // namespace nikitina_v_max_elem_matr
