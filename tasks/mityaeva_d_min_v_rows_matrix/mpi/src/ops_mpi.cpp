#include "mityaeva_d_min_v_rows_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "mityaeva_d_min_v_rows_matrix/common/include/common.hpp"

namespace mityaeva_d_min_v_rows_matrix {

MinValuesInRowsMPI::MinValuesInRowsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>{0};
}

bool MinValuesInRowsMPI::ValidationImpl() {
  const auto &input = GetInput();

  if (input.empty() || input.size() < 2) {
    return false;
  }

  int rows = input[0];
  int cols = input[1];

  if (rows <= 0 || cols <= 0) {
    return false;
  }

  size_t expected_size = 2 + (static_cast<size_t>(rows) * static_cast<size_t>(cols));
  return input.size() == expected_size;
}

bool MinValuesInRowsMPI::PreProcessingImpl() {
  return true;
}

std::vector<int> ProcessLocalRows(const std::vector<int> &input, int start_row, int my_rows, int cols) {
  std::vector<int> local_result;
  local_result.reserve(my_rows);

  for (int i = 0; i < my_rows; ++i) {
    int global_row = start_row + i;
    int row_start_index = 2 + (global_row * cols);

    if (cols == 0) {
      local_result.push_back(0);
      continue;
    }

    int min_val = input[row_start_index];
    for (int j = 1; j < cols; ++j) {
      int current_val = input[row_start_index + j];
      min_val = std::min(current_val, min_val);
    }
    local_result.push_back(min_val);
  }

  return local_result;
}

void GatherResults(int rank, int size, int rows, int rows_per_process, int remainder,
                   const std::vector<int> &local_result, std::vector<int> &output) {
  if (rank == 0) {
    std::vector<int> global_result;
    global_result.reserve(rows);

    global_result.insert(global_result.end(), local_result.begin(), local_result.end());

    for (int src = 1; src < size; ++src) {
      int src_rows = rows_per_process;
      if (src < remainder) {
        src_rows++;
      }

      if (src_rows > 0) {
        std::vector<int> recv_buffer(src_rows);
        MPI_Recv(recv_buffer.data(), src_rows, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        global_result.insert(global_result.end(), recv_buffer.begin(), recv_buffer.end());
      }
    }

    output.clear();
    output.reserve(rows + 1);
    output.push_back(rows);
    output.insert(output.end(), global_result.begin(), global_result.end());

    int output_size = static_cast<int>(output.size());
    for (int dst = 1; dst < size; ++dst) {
      MPI_Send(output.data(), output_size, MPI_INT, dst, 0, MPI_COMM_WORLD);
    }

  } else {
    if (!local_result.empty()) {
      int local_size = static_cast<int>(local_result.size());
      MPI_Send(local_result.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    int result_size = 0;
    MPI_Status status;
    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_INT, &result_size);

    std::vector<int> recv_buffer(result_size);
    MPI_Recv(recv_buffer.data(), result_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    output = recv_buffer;
  }
}

bool MinValuesInRowsMPI::RunImpl() {
  const auto &input = GetInput();

  try {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = input[0];
    int cols = input[1];

    int rows_per_process = rows / size;
    int remainder = rows % size;

    int my_rows = rows_per_process;
    if (rank < remainder) {
      my_rows++;
    }

    int start_row = 0;
    for (int i = 0; i < rank; ++i) {
      int previous_rows = rows_per_process;
      if (i < remainder) {
        previous_rows++;
      }
      start_row += previous_rows;
    }

    std::vector<int> local_result = ProcessLocalRows(input, start_row, my_rows, cols);
    GatherResults(rank, size, rows, rows_per_process, remainder, local_result, GetOutput());

    MPI_Barrier(MPI_COMM_WORLD);
    return true;

  } catch (...) {
    return false;
  }
}

bool MinValuesInRowsMPI::PostProcessingImpl() {
  const auto &output = GetOutput();
  return !output.empty() && output[0] > 0;
}

}  // namespace mityaeva_d_min_v_rows_matrix
