#include "baldin_a_my_scatter/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <cstring>

#include "baldin_a_my_scatter/common/include/common.hpp"

namespace baldin_a_my_scatter {

BaldinAMyScatterSEQ::BaldinAMyScatterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = nullptr;
}

bool BaldinAMyScatterSEQ::ValidationImpl() {
  const auto &[sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = GetInput();

  if (sendcount <= 0 || sendcount != recvcount) {
    return false;
  }

  if (sendtype != recvtype) {
    return false;
  }

  if (sendbuf == nullptr || recvbuf == nullptr) {
    return false;
  }

  auto is_sup_type = [](MPI_Datatype type) -> bool {
    return (type == MPI_INT || type == MPI_FLOAT || type == MPI_DOUBLE);
  };

  return is_sup_type(sendtype);
}

bool BaldinAMyScatterSEQ::PreProcessingImpl() {
  return true;
}

bool BaldinAMyScatterSEQ::RunImpl() {
  auto &input = GetInput();
  const auto &[sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = input;

  size_t type_size = 0;
  if (sendtype == MPI_INT) {
    type_size = sizeof(int);
  } else if (sendtype == MPI_FLOAT) {
    type_size = sizeof(float);
  } else if (sendtype == MPI_DOUBLE) {
    type_size = sizeof(double);
  }

  size_t bytes_to_copy = static_cast<size_t>(sendcount) * type_size;

  if (recvbuf != sendbuf) {
    const char *src = static_cast<const char *>(sendbuf);
    char *dst = static_cast<char *>(recvbuf);

    for (size_t i = 0; i < bytes_to_copy; ++i) {
      dst[i] = src[i];
    }
  }

  GetOutput() = recvbuf;
  return true;
}

bool BaldinAMyScatterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_my_scatter
