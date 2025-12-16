#include "ovsyannikov_n_num_mistm_in_two_str/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "ovsyannikov_n_num_mistm_in_two_str/common/include/common.hpp"

namespace ovsyannikov_n_num_mistm_in_two_str {

OvsyannikovNNumMistmInTwoStrMPI::OvsyannikovNNumMistmInTwoStrMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool OvsyannikovNNumMistmInTwoStrMPI::ValidationImpl() {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  if (proc_rank == 0) {
    return GetInput().first.size() == GetInput().second.size();
  }
  return true;
}

bool OvsyannikovNNumMistmInTwoStrMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool OvsyannikovNNumMistmInTwoStrMPI::RunImpl() {
  int proc_rank = 0;
  int proc_num = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

  int total_len = 0;
  if (proc_rank == 0) {
    total_len = static_cast<int>(GetInput().first.size());
  }

  MPI_Bcast(&total_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_len == 0) {
    return true;
  }

  std::vector<int> counts(proc_num);
  std::vector<int> displs(proc_num);

  int tail = total_len % proc_num;
  int accum = 0;
  for (int i = 0; i < proc_num; i++) {
    counts[i] = (total_len / proc_num) + (i < tail ? 1 : 0);
    displs[i] = accum;
    accum += counts[i];
  }

  int my_count = counts[proc_rank];
  std::vector<char> local_str_1(my_count);
  std::vector<char> local_str_2(my_count);

  const char *send_buf_1 = nullptr;
  const char *send_buf_2 = nullptr;

  if (proc_rank == 0) {
    send_buf_1 = GetInput().first.data();
    send_buf_2 = GetInput().second.data();
  }

  MPI_Scatterv(send_buf_1, counts.data(), displs.data(), MPI_CHAR, local_str_1.data(), my_count, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  MPI_Scatterv(send_buf_2, counts.data(), displs.data(), MPI_CHAR, local_str_2.data(), my_count, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  int priv_err_cnt = 0;
  for (int i = 0; i < my_count; ++i) {
    if (local_str_1[i] != local_str_2[i]) {
      priv_err_cnt++;
    }
  }

  int total_err_cnt = 0;
  MPI_Allreduce(&priv_err_cnt, &total_err_cnt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = total_err_cnt;

  return true;
}

bool OvsyannikovNNumMistmInTwoStrMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ovsyannikov_n_num_mistm_in_two_str
