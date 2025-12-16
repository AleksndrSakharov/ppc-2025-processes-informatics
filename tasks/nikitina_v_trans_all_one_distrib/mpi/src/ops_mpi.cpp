#include "nikitina_v_trans_all_one_distrib/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <vector>

#include "nikitina_v_trans_all_one_distrib/common/include/common.hpp"

namespace nikitina_v_trans_all_one_distrib {

TestTaskMPI::TestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp = in;
  GetInput().swap(tmp);
}

bool TestTaskMPI::ValidationImpl() {
  return true;
}

bool TestTaskMPI::PreProcessingImpl() {
  return true;
}

bool TestTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int input_size = static_cast<int>(GetInput().size());
  int global_vec_size = input_size;
  MPI_Bcast(&global_vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (global_vec_size == 0) {
    return true;
  }

  std::vector<int> current_values = GetInput();
  if (current_values.size() != static_cast<size_t>(global_vec_size)) {
    current_values.resize(static_cast<size_t>(global_vec_size), 0);
  }

  int left_child = (2 * rank) + 1;
  int right_child = (2 * rank) + 2;
  int parent = (rank - 1) / 2;

  MPI_Status status;

  if (left_child < size) {
    std::vector<int> recv_buf(static_cast<size_t>(global_vec_size));
    MPI_Recv(recv_buf.data(), global_vec_size, MPI_INT, left_child, 0, MPI_COMM_WORLD, &status);
    std::ranges::transform(current_values, recv_buf, current_values.begin(), std::plus<>());
  }

  if (right_child < size) {
    std::vector<int> recv_buf(static_cast<size_t>(global_vec_size));
    MPI_Recv(recv_buf.data(), global_vec_size, MPI_INT, right_child, 0, MPI_COMM_WORLD, &status);
    std::ranges::transform(current_values, recv_buf, current_values.begin(), std::plus<>());
  }

  if (rank != 0) {
    MPI_Send(current_values.data(), global_vec_size, MPI_INT, parent, 0, MPI_COMM_WORLD);
  }

  if (rank != 0) {
    MPI_Recv(current_values.data(), global_vec_size, MPI_INT, parent, 1, MPI_COMM_WORLD, &status);
  }

  if (left_child < size) {
    MPI_Send(current_values.data(), global_vec_size, MPI_INT, left_child, 1, MPI_COMM_WORLD);
  }
  if (right_child < size) {
    MPI_Send(current_values.data(), global_vec_size, MPI_INT, right_child, 1, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    GetOutput().resize(static_cast<size_t>(global_vec_size));
    std::ranges::copy(current_values, GetOutput().begin());
  }

  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace nikitina_v_trans_all_one_distrib
