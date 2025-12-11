#include "nikitina_v_max_elem_matr/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <vector>

#include "nikitina_v_max_elem_matr/common/include/common.hpp"

namespace nikitina_v_max_elem_matr {

MaxElementMatrMPI::MaxElementMatrMPI(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaxElementMatrMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &in = GetInput();
    if (in.size() < 2) {
      return false;
    }
    rows_ = in[0];
    cols_ = in[1];
    return rows_ >= 0 && cols_ >= 0 && static_cast<size_t>(rows_) * cols_ == in.size() - 2;
  }
  return true;
}

bool MaxElementMatrMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &in = GetInput();
    matrix_.clear();
    if (rows_ > 0 && cols_ > 0) {
      matrix_.reserve(static_cast<size_t>(rows_) * cols_);
      std::copy(in.begin() + 2, in.end(), std::back_inserter(matrix_));
    }
  }
  return true;
}

void MaxElementMatrMPI::CalculateScatterParams(int total_elements, int world_size, std::vector<int> &sendcounts,
                                               std::vector<int> &displs) {
  const int elements_per_proc = total_elements / world_size;
  const int remainder_elements = total_elements % world_size;
  int current_displ = 0;
  for (int i = 0; i < world_size; ++i) {
    sendcounts[i] = (i < remainder_elements) ? elements_per_proc + 1 : elements_per_proc;
    displs[i] = current_displ;
    current_displ += sendcounts[i];
  }
}

int MaxElementMatrMPI::FindLocalMax(const std::vector<int> &data) {
  if (data.empty()) {
    return std::numeric_limits<int>::min();
  }
  int max_val = data[0];
  for (size_t i = 1; i < data.size(); ++i) {
    max_val = std::max(max_val, data[i]);
  }
  return max_val;
}

bool MaxElementMatrMPI::RunImpl() {
  int world_size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Bcast(&rows_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (static_cast<size_t>(rows_) * cols_ == 0) {
    global_max_ = std::numeric_limits<int>::min();
  } else {
    const int total_elements = rows_ * cols_;
    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);

    if (rank == 0) {
      CalculateScatterParams(total_elements, world_size, sendcounts, displs);
    }

    MPI_Bcast(sendcounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> recv_buf(sendcounts[rank]);
    const int *send_buf = (rank == 0) ? matrix_.data() : nullptr;

    MPI_Scatterv(send_buf, sendcounts.data(), displs.data(), MPI_INT, recv_buf.data(), sendcounts[rank], MPI_INT, 0,
                 MPI_COMM_WORLD);

    int local_max = FindLocalMax(recv_buf);
    MPI_Reduce(&local_max, &global_max_, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  }

  MPI_Bcast(&global_max_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}

bool MaxElementMatrMPI::PostProcessingImpl() {
  GetOutput() = global_max_;
  return true;
}

}  // namespace nikitina_v_max_elem_matr
