#include "gonozov_l_elem_vec_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "gonozov_l_elem_vec_sum/common/include/common.hpp"

namespace gonozov_l_elem_vec_sum {

GonozovLElemVecSumMPI::GonozovLElemVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GonozovLElemVecSumMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool GonozovLElemVecSumMPI::PreProcessingImpl() {
  vector_size_ = static_cast<int>(GetInput().size());
  return true;
}

bool GonozovLElemVecSumMPI::RunImpl() {
  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  if (vector_size_ < proc_num) {
    OutType result = 0;
    if (proc_rank == 0) {
      for (int i = 0; i < vector_size_; i++) {
        result += GetInput()[i];
      }
    }
    MPI_Bcast(&result, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    GetOutput() = result;
    return true;
  }

  int n = vector_size_ / proc_num;
  int remainder = vector_size_ % proc_num;
  int local_size = n + (proc_rank < remainder ? 1 : 0);

  std::vector<int> sendcounts(proc_num);
  std::vector<int> displs(proc_num);
  std::vector<int> subvector(local_size);

  if (proc_rank == 0) {
    int offset = 0;
    for (int i = 0; i < proc_num; i++) {
      sendcounts[i] = n + (i < remainder ? 1 : 0);
      displs[i] = offset;
      offset += sendcounts[i];
    }
  }

  MPI_Scatterv((proc_rank == 0) ? GetInput().data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
               subvector.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

  OutType local_sum = std::accumulate(subvector.begin(), subvector.end(), 0LL);

  OutType global_sum = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = global_sum;

  return true;
}

bool GonozovLElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gonozov_l_elem_vec_sum
