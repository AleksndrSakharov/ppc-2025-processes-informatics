#include "levonychev_i_min_val_rows_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"

namespace levonychev_i_min_val_rows_matrix {

LevonychevIMinValRowsMatrixMPI::LevonychevIMinValRowsMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
  GetOutput() = {};
}
bool LevonychevIMinValRowsMatrixMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const size_t vector_size = std::get<0>(GetInput()).size();
    const int rows = std::get<1>(GetInput());
    const int cols = std::get<2>(GetInput());
    return vector_size != 0 && rows != 0 && cols != 0 &&
           (vector_size == static_cast<size_t>(rows) * static_cast<size_t>(cols));
  }
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::PreProcessingImpl() {
  int rank = 0;
  int rows = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    rows = std::get<1>(GetInput());
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput().resize(rows);
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::RunImpl() {
  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  OutType &global_min_values = GetOutput();

  int rows = 0;
  int cols = 0;
  std::vector<int> recvcounts_scatterv(proc_num);
  std::vector<int> displs_scatterv(proc_num);
  std::vector<int> recvcounts_gatherv(proc_num);
  std::vector<int> displs_gatherv(proc_num);

  if (proc_rank == 0) {
    rows = std::get<1>(GetInput());
    cols = std::get<2>(GetInput());

    for (int i = 0; i < proc_num; ++i) {
      int local_count_of_rows = i == (proc_num - 1) ? ((rows / proc_num) + (rows % proc_num)) : (rows / proc_num);
      recvcounts_scatterv[i] = local_count_of_rows * cols;
      recvcounts_gatherv[i] = local_count_of_rows;
      int start = i * (rows / proc_num);
      displs_scatterv[i] = start * cols;
      displs_gatherv[i] = start;
    }
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int local_count_of_rows = (proc_rank == (proc_num - 1)) ? ((rows / proc_num) + (rows % proc_num)) : (rows / proc_num);
  int recvcount = local_count_of_rows * cols;
  OutType local_matrix(recvcount);
  MPI_Scatterv(std::get<0>(GetInput()).data(), recvcounts_scatterv.data(), displs_scatterv.data(), MPI_INT,
               local_matrix.data(), recvcount, MPI_INT, 0, MPI_COMM_WORLD);

  OutType local_min_values(local_count_of_rows);

  for (int i = 0; i < local_count_of_rows; ++i) {
    const int start = cols * i;
    int min_value = local_matrix[start];
    for (int j = 1; j < cols; ++j) {
      min_value = std::min(local_matrix[start + j], min_value);
    }
    local_min_values[i] = min_value;
  }

  MPI_Gatherv(local_min_values.data(), local_count_of_rows, MPI_INT, global_min_values.data(),
              recvcounts_gatherv.data(), displs_gatherv.data(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(global_min_values.data(), rows, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_min_val_rows_matrix
