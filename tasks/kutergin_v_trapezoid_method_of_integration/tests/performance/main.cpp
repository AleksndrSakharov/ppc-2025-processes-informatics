#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <fstream>
#include <stdexcept>

#include "../../common/include/common.hpp"
#include "../../mpi/include/trapezoid_integration_mpi.hpp"
#include "../../seq/include/trapezoid_integration_sequential.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace kutergin_v_trapezoid_seq {

class KuterginVRunPerfTests
    : public ppc::util::BaseRunPerfTests<InType, OutType>  // наследник ppc::util::BaseRunPerfTests<InType, OutType>
{
 protected:
  // подготовка входных данных перед каждым тестом
  void SetUp() override {
    const auto path_to_input =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_kutergin_v_trapezoid_method_of_integration,
                                       "input.txt");  // получение абсолютного пути к файлу с данными

    std::ifstream file(path_to_input);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + path_to_input);
    }
    file >> input_data_.a >> input_data_.b >> input_data_.n;
    file.close();
  }

  // проверка результата после выполнения задачи
  bool CheckTestOutputData(OutType &output_data) final {
    int process_rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    }
    if (process_rank == 0)  // проверку выполняет только нулевой процесс
    {
      const double expected =
          (input_data_.b * input_data_.b * input_data_.b / 3.0) - (input_data_.a * input_data_.a * input_data_.a / 3.0);
      return std::abs(output_data - expected) < 1e-5;
    }
    return true;  // у всех остальных процессов все в порядке
  }

  // возврат подготовленных входных данных для задачи
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(KuterginVRunPerfTests, PerfTest)  // параметризованный тест
{
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, kutergin_v_trapezoid_mpi::TrapezoidIntegrationMPI,
                                                       TrapezoidIntegrationSequential>(
    PPC_SETTINGS_kutergin_v_trapezoid_method_of_integration);

// NOLINTNEXTLINE(modernize-type-traits, cppcoreguidelines-avoid-non-const-global-variables)
INSTANTIATE_TEST_SUITE_P(TrapezoidPerf, KuterginVRunPerfTests, ppc::util::TupleToGTestValues(kAllPerfTasks),
                         KuterginVRunPerfTests::CustomPerfTestName);

}  // namespace

}  // namespace kutergin_v_trapezoid_seq
