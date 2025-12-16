#include "baldin_a_my_scatter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "baldin_a_my_scatter/common/include/common.hpp"

namespace baldin_a_my_scatter {

BaldinAMyScatterMPI::BaldinAMyScatterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BaldinAMyScatterMPI::ValidationImpl() {
  const auto &[sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = GetInput();

  if (sendcount <= 0 || sendcount != recvcount || root < 0) {
    return false;
  }
  if (sendtype != recvtype) {
    return false;
  }

  auto is_sup_type = [](MPI_Datatype type) -> bool {
    return (type == MPI_INT || type == MPI_FLOAT || type == MPI_DOUBLE);
  };

  return is_sup_type(sendtype);
}

bool BaldinAMyScatterMPI::PreProcessingImpl() {
  int root = std::get<6>(GetInput());

  int world_size = 0;
  MPI_Comm_size(std::get<7>(GetInput()), &world_size);

  // если root выходит за границы, корректируем его
  if (root >= world_size) {
    std::get<6>(GetInput()) = root % world_size;
  }

  return true;
}

namespace {

MPI_Aint GetDataTypeExtent(MPI_Datatype type) {
  MPI_Aint lb = 0;
  MPI_Aint extent = 0;
  MPI_Type_get_extent(type, &lb, &extent);
  return extent;
}

int VirtualToRealRank(int v_rank, int root, int size) {
  return (v_rank + root) % size;
}

int CalculateSubtreeSize(int v_dest, int mask, int size) {
  return std::min(v_dest + mask, size);
}

int CalculateInitialMask(int size) {
  int mask = 1;
  while (mask < size) {
    mask <<= 1;
  }
  return mask >> 1;
}

void PrepareRootBuffer(const void *sendbuf, int size, int root, int count, MPI_Aint extent, std::vector<char> &buffer) {
  size_t total_bytes = static_cast<size_t>(size) * count * extent;
  size_t chunk_bytes = static_cast<size_t>(count) * extent;

  buffer.resize(total_bytes);

  const char *send_ptr = static_cast<const char *>(sendbuf);
  char *tmp_ptr = buffer.data();

  // Логика сдвига: [root...end] -> начало, [0...root] -> конец
  size_t first_part_bytes = (size - root) * chunk_bytes;
  size_t second_part_bytes = root * chunk_bytes;

  std::memcpy(tmp_ptr, send_ptr + second_part_bytes, first_part_bytes);
  std::memcpy(tmp_ptr + first_part_bytes, send_ptr, second_part_bytes);
}

}  // namespace

bool BaldinAMyScatterMPI::RunImpl() {
  auto &input = GetInput();
  const auto &[sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = input;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_Aint extent = GetDataTypeExtent(rank == root ? sendtype : recvtype);

  std::vector<char> temp_buffer;
  const char *curr_buf_ptr = nullptr;

  // --- ЭТАП 1: Подготовка (только на root) ---
  if (rank == root) {
    PrepareRootBuffer(sendbuf, size, root, sendcount, extent, temp_buffer);
    curr_buf_ptr = temp_buffer.data();
  }

  int v_rank = (rank - root + size) % size;
  int mask = CalculateInitialMask(size);

  // --- ЭТАП 2: Рассылка по дереву ---
  while (mask > 0) {
    // Если процесс - отправитель на этом уровне
    if (v_rank % (2 * mask) == 0) {
      int v_dest = v_rank + mask;

      if (v_dest < size) {
        int subtree_size = CalculateSubtreeSize(v_dest, mask, size);
        int count_to_send = (subtree_size - v_dest) * recvcount;

        size_t offset_bytes = static_cast<size_t>(v_dest - v_rank) * recvcount * extent;
        int real_dest = VirtualToRealRank(v_dest, root, size);

        MPI_Send(curr_buf_ptr + offset_bytes, count_to_send, (rank == root ? sendtype : recvtype), real_dest, 0, comm);
      }
    }

    // Если процесс - получатель
    else if (v_rank % (2 * mask) == mask) {
      int v_source = v_rank - mask;
      int real_source = VirtualToRealRank(v_source, root, size);

      int subtree_end = CalculateSubtreeSize(v_rank, mask, size);
      int count_to_recv = (subtree_end - v_rank) * recvcount;

      size_t bytes_to_recv = static_cast<size_t>(count_to_recv) * extent;
      temp_buffer.resize(bytes_to_recv);

      MPI_Recv(temp_buffer.data(), count_to_recv, recvtype, real_source, 0, comm, MPI_STATUS_IGNORE);

      // Теперь работаем с полученным буфером
      curr_buf_ptr = temp_buffer.data();
    }

    mask >>= 1;
  }

  // --- ЭТАП 3: Копирование в пользовательский буфер ---
  if (recvbuf != MPI_IN_PLACE && curr_buf_ptr != nullptr) {
    // Копируем только свою долю (recvcount)
    std::memcpy(recvbuf, curr_buf_ptr, recvcount * extent);
  }
  GetOutput() = recvbuf;
  return true;
}

bool BaldinAMyScatterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_my_scatter
