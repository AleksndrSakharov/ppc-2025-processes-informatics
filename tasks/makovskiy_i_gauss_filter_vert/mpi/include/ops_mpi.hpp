#pragma once

#include <task/include/task.hpp>
#include <vector>

#include "makovskiy_i_gauss_filter_vert/common/include/common.hpp"

namespace makovskiy_i_gauss_filter_vert {

class GaussFilterMPI : public BaseTask {
 public:
  explicit GaussFilterMPI(const InType &in);

  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

 protected:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void ScatterData(int rank, int world_size);
  void ScatterDataRoot(int world_size);
  void ScatterDataLeaf();

  std::vector<int> ComputeLocal(int rank, int world_size);

  void GatherData(int rank, int world_size, const std::vector<int> &local_output);
  void GatherDataRoot(int world_size, std::vector<int> &final_output, const std::vector<int> &local_output) const;
  void GatherDataLeaf(const std::vector<int> &local_output) const;

  std::vector<int> local_strip_;
  int strip_width_ = 0;
  int total_width_ = 0;
  int total_height_ = 0;
};

}  // namespace makovskiy_i_gauss_filter_vert
