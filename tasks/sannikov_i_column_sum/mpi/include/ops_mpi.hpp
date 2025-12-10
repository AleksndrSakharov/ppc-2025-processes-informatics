#pragma once

#include <cstdint>
#include <vector>

#include "sannikov_i_column_sum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sannikov_i_column_sum {

class SannikovIColumnSumMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SannikovIColumnSumMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void PrepareSendBuffer(const InType &input_matrix, int rank, std::uint64_t rows, std::uint64_t columns,
                                std::vector<int> &sendbuf);
};

}  // namespace sannikov_i_column_sum
