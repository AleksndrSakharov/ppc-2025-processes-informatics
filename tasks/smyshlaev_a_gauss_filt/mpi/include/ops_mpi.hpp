#pragma once

#include <cstdint>
#include <vector>

#include "smyshlaev_a_gauss_filt/common/include/common.hpp"
#include "task/include/task.hpp"

namespace smyshlaev_a_gauss_filt {

class SmyshlaevAGaussFiltMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SmyshlaevAGaussFiltMPI(const InType &in);

 private:
  struct BlockInfo {
    int start_row;
    int start_col;
    int block_height;
    int block_width;
    int padded_height;
    int padded_width;
    int count;
  };

  struct DecompositionInfo {
    int grid_rows = 0;
    int grid_cols = 0;
    std::vector<BlockInfo> blocks;
    std::vector<int> sendcounts;
    std::vector<int> displs;
  };

  bool RunSequential();
  void BroadcastImageDimensions(int &width, int &height, int &channels);
  static void SetupDecomposition(DecompositionInfo &info, int width, int height, int channels);
  std::vector<uint8_t> PrepareScatterBuffer(const DecompositionInfo &info, int width, int height, int channels);
  std::vector<uint8_t> ProcessLocalBlock(const DecompositionInfo &info, int width, int height, int channels);
  void CollectResult(const std::vector<uint8_t> &local_result, const DecompositionInfo &info, int width, int height,
                     int channels);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace smyshlaev_a_gauss_filt
