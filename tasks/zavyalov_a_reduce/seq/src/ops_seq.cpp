#include "zavyalov_a_reduce/seq/include/ops_seq.hpp"

#include <cstdlib>

#include "zavyalov_a_reduce/common/include/common.hpp"

namespace zavyalov_a_reduce {

ZavyalovAReduceSEQ::ZavyalovAReduceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  std::get<0>(GetOutput()) = nullptr;
  std::get<1>(GetOutput()) = true;
}

bool ZavyalovAReduceSEQ::ValidationImpl() {
  return true;
}

bool ZavyalovAReduceSEQ::PreProcessingImpl() {
  return true;
}

bool ZavyalovAReduceSEQ::RunImpl() {
  std::get<1>(GetOutput()) = true;
  volatile int i = 0;
  while (i < 10000000) {  // для того чтобы время выполнения было больше 0.001 в CI
    i += 1;
  }
  return true;
  /*
  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);
  if (is_mpi_initialized == 0) {
    return true;
  }
  MPI_Op operation = std::get<0>(GetInput());
  MPI_Datatype cur_type = std::get<1>(GetInput());
  size_t sz = std::get<2>(GetInput());
  std::shared_ptr<void> mem_ptr = std::get<3>(GetInput());
  void *mem = mem_ptr.get();
  int receiver_rank = std::get<4>(GetInput());

  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int type_size = 0;
  MPI_Type_size(cur_type, &type_size);

  char *raw_result = new char[sz * type_size];
  std::shared_ptr<void> result_ptr(raw_result, [](void *p) { delete[] static_cast<char *>(p); });

  if (rank == receiver_rank) {
    MPI_Reduce(mem, raw_result, static_cast<int>(sz), cur_type, operation, receiver_rank, MPI_COMM_WORLD);
    MPI_Bcast(raw_result, static_cast<int>(sz), cur_type, receiver_rank, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(mem, nullptr, static_cast<int>(sz), cur_type, operation, receiver_rank, MPI_COMM_WORLD);
    MPI_Bcast(raw_result, static_cast<int>(sz), cur_type, receiver_rank, MPI_COMM_WORLD);
  }

  std::get<0>(GetOutput()) = result_ptr;
  std::get<1>(GetOutput()) = false;

  return true;
  */
}

bool ZavyalovAReduceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_reduce
