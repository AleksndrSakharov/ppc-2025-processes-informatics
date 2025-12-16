#include "romanova_v_min_by_matrix_rows_processes/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "romanova_v_min_by_matrix_rows_processes/common/include/common.hpp"

namespace romanova_v_min_by_matrix_rows_processes {

RomanovaVMinByMatrixRowsMPI::RomanovaVMinByMatrixRowsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType(in.size());
}

bool RomanovaVMinByMatrixRowsMPI::ValidationImpl() {
  bool status = false;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    status = !GetInput().empty() && !GetInput()[0].empty();
  }
  MPI_Bcast(&status, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return status;
}

bool RomanovaVMinByMatrixRowsMPI::PreProcessingImpl() {
  in_data_ = GetInput();
  return true;
}

bool RomanovaVMinByMatrixRowsMPI::RunImpl() {
  int n = 0;
  int rank = 0;
  int delta = 0;
  int extra = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &n);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> recv_counts(n);
  std::vector<int> send_counts(n);

  std::vector<int> displs_scatt(n);
  std::vector<int> displs_gath(n);

  OutType flat_data;

  if (rank == 0) {
    n_ = in_data_.size();
    m_ = ((!in_data_.empty()) ? in_data_[0].size() : 0);

    delta = static_cast<int>(n_ / n);
    extra = static_cast<int>(n_ % n);

    recv_counts = std::vector<int>(n, delta);
    recv_counts[n - 1] += extra;

    send_counts = std::vector<int>(n, static_cast<int>(delta * m_));
    send_counts[n - 1] += static_cast<int>(extra * m_);

    for (int i = 1; i < n; i++) {
      displs_gath[i] = displs_gath[i - 1] + delta;
      displs_scatt[i] = displs_scatt[i - 1] + static_cast<int>(delta * m_);
    }

    for (const auto &vec : in_data_) {
      flat_data.insert(flat_data.end(), vec.begin(), vec.end());
    }
  }

  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(&delta, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&extra, 1, MPI_INT, 0, MPI_COMM_WORLD);

  OutType local_data((delta + (rank == n - 1 ? extra : 0)) * m_);

  MPI_Scatterv(rank == 0 ? flat_data.data() : nullptr, send_counts.data(), displs_scatt.data(), MPI_INT,
               local_data.data(), static_cast<int>(local_data.size()), MPI_INT, 0, MPI_COMM_WORLD);

  OutType temp(delta + (rank == n - 1 ? extra : 0));

  for (size_t i = 0; i < temp.size(); i++) {
    temp[i] = local_data[i * m_];
    for (size_t j = 1; j < m_; j++) {
      temp[i] = std::min(temp[i], local_data[(i * m_) + j]);
    }
  }

  res_ = OutType(n_);

  MPI_Gatherv(temp.data(), static_cast<int>(temp.size()), MPI_INT, res_.data(), recv_counts.data(), displs_gath.data(),
              MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(res_.data(), static_cast<int>(n_), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool RomanovaVMinByMatrixRowsMPI::PostProcessingImpl() {
  GetOutput() = res_;
  return true;
}

}  // namespace romanova_v_min_by_matrix_rows_processes
