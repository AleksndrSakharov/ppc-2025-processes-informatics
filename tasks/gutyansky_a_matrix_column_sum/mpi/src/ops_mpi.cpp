#include "gutyansky_a_matrix_column_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "gutyansky_a_matrix_column_sum/common/include/common.hpp"

namespace gutyansky_a_matrix_column_sum {

GutyanskyAMatrixColumnSumMPI::GutyanskyAMatrixColumnSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput() = in;
  GetOutput() = {};
}

bool GutyanskyAMatrixColumnSumMPI::ValidationImpl() {
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return GetInput().IsValid();
  }

  return true;
}

bool GutyanskyAMatrixColumnSumMPI::PreProcessingImpl() {
  return true;
}

bool GutyanskyAMatrixColumnSumMPI::RunImpl() {
  int rank = -1;
  int p_count = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p_count);

  size_t row_count = GetInput().rows;
  size_t col_count = GetInput().cols;

  MPI_Bcast(&row_count, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&col_count, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  size_t chunk_size = row_count / static_cast<size_t>(p_count);
  size_t remainder_size = row_count % static_cast<size_t>(p_count);

  size_t rows_count = chunk_size + (std::cmp_less(rank, remainder_size) ? 1 : 0);
  size_t elements_count = col_count * rows_count;

  std::vector<int32_t> input_data_chunk(elements_count);
  std::vector<int32_t> partial_res(col_count, static_cast<int32_t>(0));

  if (rank == 0) {
    std::vector<int> send_counts(p_count);
    std::vector<int> locs(p_count);

    for (int i = 0; i < p_count; i++) {
      size_t rows_for_proc = chunk_size + (std::cmp_less(i, remainder_size) ? 1 : 0);
      send_counts[i] = static_cast<int>(rows_for_proc * col_count);
    }

    for (int i = 1; i < p_count; i++) {
      locs[i] = locs[i - 1] + send_counts[i - 1];
    }

    MPI_Scatterv(GetInput().data.data(), send_counts.data(), locs.data(), MPI_INTEGER4, input_data_chunk.data(),
                 static_cast<int>(elements_count), MPI_INTEGER4, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INTEGER4, input_data_chunk.data(), static_cast<int>(elements_count),
                 MPI_INTEGER4, 0, MPI_COMM_WORLD);
  }

  for (size_t i = 0; i < rows_count; i++) {
    for (size_t j = 0; j < col_count; j++) {
      partial_res[j] += input_data_chunk[(i * col_count) + j];
    }
  }

  if (rank == 0) {
    GetOutput().resize(col_count);
  }

  MPI_Reduce(partial_res.data(), GetOutput().data(), static_cast<int>(col_count), MPI_INTEGER4, MPI_SUM, 0,
             MPI_COMM_WORLD);

  return true;
}

bool GutyanskyAMatrixColumnSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gutyansky_a_matrix_column_sum
