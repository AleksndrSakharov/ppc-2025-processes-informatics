#include "lukin_i_elem_vec_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "lukin_i_elem_vec_sum/common/include/common.hpp"

namespace lukin_i_elem_vec_sum {

LukinIElemVecSumMPI::LukinIElemVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }

  GetOutput() = 0;
}

bool LukinIElemVecSumMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return (static_cast<int>(GetInput().size()) != 0);
  }
  return true;
}

bool LukinIElemVecSumMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    vec_size_ = static_cast<int>(GetInput().size());
  }
  return true;
}

bool LukinIElemVecSumMPI::RunImpl() {
  int proc_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Bcast(&vec_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (proc_count > vec_size_) {
    if (rank == 0) {
      GetOutput() = std::accumulate(GetInput().begin(), GetInput().end(), 0LL);
    } else {
      GetOutput() = 0;
    }
    OutType output = GetOutput();
    MPI_Bcast(&output, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    GetOutput() = output;
    return true;
  }

  std::vector<int> sendcounts(proc_count);
  std::vector<int> offsets(proc_count);

  const int part = vec_size_ / proc_count;
  const int reminder = vec_size_ % proc_count;

  int offset = 0;
  for (int i = 0; i < proc_count; i++) {
    sendcounts[i] = part + (i < reminder ? 1 : 0);
    offsets[i] = offset;
    offset += sendcounts[i];
  }

  int local_size = sendcounts[rank];
  std::vector<int> local_vec(local_size);

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, sendcounts.data(), offsets.data(), MPI_INT, local_vec.data(),
               local_size, MPI_INT, 0, MPI_COMM_WORLD);

  OutType local_sum = std::accumulate(local_vec.begin(), local_vec.end(), 0LL);

  OutType global_sum = 0LL;

  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_sum;

  return true;
}

bool LukinIElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_elem_vec_sum
