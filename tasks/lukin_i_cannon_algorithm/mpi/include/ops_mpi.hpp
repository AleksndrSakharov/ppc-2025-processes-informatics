#pragma once

#include <mpi.h>

#include "lukin_i_cannon_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lukin_i_cannon_algorithm {

class LukinICannonAlgorithmMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LukinICannonAlgorithmMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void MulNSum(const double *a, const double *b, double *c, int size);

  bool RunSeq(int global_rank);
  bool RunCannon(int global_rank, int proc_count);
  void CannonWorkers(MPI_Comm comm, int grid_size, int working_proc_count);
  void CannonNonWorkers();
  void MatrixPack(double *a_blocks, double *b_blocks, int working_proc_count, int block_elems, int block_size,
                  int grid_size);
  void MatrixUnpack(double *c, const double *c_blocks, int working_proc_count, int block_elems, int block_size,
                    int grid_size) const;

  int size_ = 0;
};

}  // namespace lukin_i_cannon_algorithm
