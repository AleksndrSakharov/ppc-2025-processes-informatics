#include "luzan_e_matrix_rows_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "luzan_e_matrix_rows_sum/common/include/common.hpp"

namespace luzan_e_matrix_rows_sum {

LuzanEMatrixRowsSumMPI::LuzanEMatrixRowsSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = {};

  // saving matrix only if it's rank=0
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetInput() = in;
  } else {
    GetInput() = {};
  }
}

bool LuzanEMatrixRowsSumMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }

  int height = std::get<1>(GetInput());
  int width = std::get<2>(GetInput());
  return std::get<0>(GetInput()).size() == static_cast<size_t>(height) * static_cast<size_t>(width) && height > 0 &&
         width > 0;
}

bool LuzanEMatrixRowsSumMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }

  int height = std::get<1>(GetInput());
  GetOutput().resize(height);
  for (int row = 0; row < height; row++) {
    GetOutput()[row] = 0;
  }
  return true;
}

bool LuzanEMatrixRowsSumMPI::RunImpl() {
  // mpi things
  int height = 0;
  int width = 0;
  std::tuple_element_t<0, InType> mat;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // getting input matrix on rank=0
  // getting & sharing matrix sizes
  std::vector<int> dim(2, 0);
  if (rank == 0) {
    mat = std::get<0>(GetInput());
    height = std::get<1>(GetInput());
    width = std::get<2>(GetInput());
  }
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // calcilating shifts & rows_per_proc (only about rows rigth now)
  int rest = height % size;
  std::vector<int> shift(size, 0);
  std::vector<int> per_proc(size, height / size);  // rows per proc

  int accumulator = 0;
  for (int i = 0; i < size; i++) {
    if (rest > 0) {
      per_proc[i]++;
      rest--;
    }
    shift[i] = accumulator;
    accumulator = per_proc[i] + shift[i];
  }

  // preparing to recieve data
  std::vector<int> recv(static_cast<size_t>(per_proc[rank] * width));

  for (int i = 0; i < size; i++) {
    per_proc[i] *= width;  // now it's about elements
    shift[i] *= width;
  }
  MPI_Scatterv(mat.data(), per_proc.data(), shift.data(), MPI_INT, recv.data(), per_proc[rank], MPI_INT, 0,
               MPI_COMM_WORLD);
  mat.clear();  // no need anymore

  // calculating
  std::vector<int> rows_sum(static_cast<size_t>(per_proc[rank] / width));  // sums
  int rows_to_calc = static_cast<int>(per_proc[rank] / width);
  for (int row = 0; row < rows_to_calc; row++) {
    for (int col = 0; col < width; col++) {
      rows_sum[row] += recv[(row * width) + col];
    }
  }

  for (int i = 0; i < size; i++) {
    per_proc[i] /= width;  // back to rows
    shift[i] /= width;
  }

  std::vector<int> fin_sum(height);
  MPI_Gatherv(rows_sum.data(), rows_to_calc, MPI_INT, fin_sum.data(), per_proc.data(), shift.data(), MPI_INT, 0,
              MPI_COMM_WORLD);
  MPI_Bcast(fin_sum.data(), height, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = fin_sum;
  return true;
}

bool LuzanEMatrixRowsSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace luzan_e_matrix_rows_sum
