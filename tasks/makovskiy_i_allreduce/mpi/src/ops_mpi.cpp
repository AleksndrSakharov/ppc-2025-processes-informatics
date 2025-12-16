#include "makovskiy_i_allreduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "makovskiy_i_allreduce/common/include/common.hpp"

namespace makovskiy_i_allreduce {

TestTaskMPI::TestTaskMPI(const InType &in) {
  InType temp(in);
  this->GetInput().swap(temp);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool TestTaskMPI::ValidationImpl() {
  return true;
}

bool TestTaskMPI::PreProcessingImpl() {
  this->GetOutput().resize(1);
  return true;
}

bool TestTaskMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(this->GetInput().size());
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    this->GetOutput()[0] = 0;
    return true;
  }

  int delta = total_size / world_size;
  int remainder = total_size % world_size;

  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);

  if (rank == 0) {
    int current_displ = 0;
    for (int i = 0; i < world_size; ++i) {
      sendcounts[i] = delta + (i < remainder ? 1 : 0);
      displs[i] = current_displ;
      current_displ += sendcounts[i];
    }
  }

  int my_count = 0;
  MPI_Scatter(sendcounts.data(), 1, MPI_INT, &my_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  local_input_.resize(my_count);

  MPI_Scatterv(rank == 0 ? this->GetInput().data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
               local_input_.data(), my_count, MPI_INT, 0, MPI_COMM_WORLD);

  int local_sum = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  int global_sum = 0;

  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  this->GetOutput()[0] = global_sum;

  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace makovskiy_i_allreduce
