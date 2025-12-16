#pragma once

#include <mpi.h>

#include <cstring>
#include <memory>
#include <tuple>

#include "task/include/task.hpp"

namespace zavyalov_a_reduce {
// InType: операция, тип данных, число элементов в передаваемом массиве, указатель на память где хранится массив, номер
// процесса-получателя
using InType = std::tuple<MPI_Op, MPI_Datatype, size_t, std::shared_ptr<void>,
                          int>;  // void* instead of vector because input type can differ
using OutType =
    std::tuple<std::shared_ptr<void>, bool>;  // result of mpi_reduce(void* instead of vector because input type can
                                              // differ), bool - tells if it is seq verison or mpi. true -> seq
using TestType = std::tuple<MPI_Op, MPI_Datatype, size_t,
                            int>;  // operation type, vector elements type, size of vectors, receiver process rank
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zavyalov_a_reduce
