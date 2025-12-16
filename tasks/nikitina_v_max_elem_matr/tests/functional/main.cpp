#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <tuple>

#include "nikitina_v_max_elem_matr/common/include/common.hpp"
#include "nikitina_v_max_elem_matr/mpi/include/ops_mpi.hpp"
#include "nikitina_v_max_elem_matr/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nikitina_v_max_elem_matr {

using TestType = std::tuple<int, int, int>;

class NikitinaVMaxElemMatrFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<ParamType> &info) {
    auto task_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(info.param);
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(info.param);
    std::string tech = "unknown";
    if (task_name.find("seq") != std::string::npos) {
      tech = "seq";
    }
    if (task_name.find("mpi") != std::string::npos) {
      tech = "mpi";
    }
    std::string test_name = "tech_" + tech + "_test_id_" + std::to_string(std::get<0>(params)) + "_rows_" +
                            std::to_string(std::get<1>(params)) + "_cols_" + std::to_string(std::get<2>(params));
    return test_name;
  }

 protected:
  void SetUp() override {
    auto task_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());
    bool is_mpi_test = task_name.find("mpi") != std::string::npos;

    int rank = 0;
    if (is_mpi_test) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank == 0) {
      TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
      int rows = std::get<1>(params);
      int cols = std::get<2>(params);

      if (rows <= 0 || cols <= 0) {
        input_data_ = {rows, cols};
        expected_output_ = std::numeric_limits<int>::min();
      } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(-1000, 1000);
        int max_val = std::numeric_limits<int>::min();

        InType generated_matr(2 + (static_cast<size_t>(rows) * cols));
        generated_matr[0] = rows;
        generated_matr[1] = cols;

        for (int i = 0; i < rows * cols; ++i) {
          generated_matr[i + 2] = distrib(gen);
          if (i == 0 || generated_matr[i + 2] > max_val) {
            max_val = generated_matr[i + 2];
          }
        }
        input_data_ = generated_matr;
        expected_output_ = max_val;
      }
    }

    if (is_mpi_test) {
      int input_size = static_cast<int>(input_data_.size());
      MPI_Bcast(&input_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

      if (rank != 0) {
        input_data_.resize(input_size);
      }

      MPI_Bcast(input_data_.data(), input_size, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&expected_output_, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return (output_data == expected_output_);
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_{};
};

namespace {

TEST_P(NikitinaVMaxElemMatrFuncTests, FindMaxElement) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(1, 10, 10), std::make_tuple(2, 5, 15), std::make_tuple(3, 1, 30), std::make_tuple(4, 30, 1),
    std::make_tuple(5, 1, 1),   std::make_tuple(6, 0, 10), std::make_tuple(7, 10, 0), std::make_tuple(8, 0, 0)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MaxElementMatrSEQ, InType>(kTestParam, PPC_SETTINGS_nikitina_v_max_elem_matr),
    ppc::util::AddFuncTask<MaxElementMatrMPI, InType>(kTestParam, PPC_SETTINGS_nikitina_v_max_elem_matr));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(NikitinaV_MaxElementMatr_Func, NikitinaVMaxElemMatrFuncTests, kGtestValues,
                         NikitinaVMaxElemMatrFuncTests::PrintTestParam);

}  // namespace

TEST(NikitinaVMaxElemMatrValidation, FailsOnEmptyInput) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    auto *seq_task = new MaxElementMatrSEQ({});
    ASSERT_FALSE(seq_task->Validation());
  }
  auto *mpi_task = new MaxElementMatrMPI({});
  if (rank == 0) {
    ASSERT_FALSE(mpi_task->Validation());
  } else {
    mpi_task->Validation();
  }
}

TEST(NikitinaVMaxElemMatrValidation, FailsOnTooSmallInput) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    auto *seq_task = new MaxElementMatrSEQ({1});
    ASSERT_FALSE(seq_task->Validation());
  }
  auto *mpi_task = new MaxElementMatrMPI({1});
  if (rank == 0) {
    ASSERT_FALSE(mpi_task->Validation());
  } else {
    mpi_task->Validation();
  }
}

TEST(NikitinaVMaxElemMatrValidation, FailsOnNegativeRows) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    auto *seq_task = new MaxElementMatrSEQ({-1, 5});
    ASSERT_FALSE(seq_task->Validation());
  }
  auto *mpi_task = new MaxElementMatrMPI({-1, 5});
  if (rank == 0) {
    ASSERT_FALSE(mpi_task->Validation());
  } else {
    mpi_task->Validation();
  }
}

TEST(NikitinaVMaxElemMatrValidation, FailsOnNegativeCols) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    auto *seq_task = new MaxElementMatrSEQ({5, -1});
    ASSERT_FALSE(seq_task->Validation());
  }
  auto *mpi_task = new MaxElementMatrMPI({5, -1});
  if (rank == 0) {
    ASSERT_FALSE(mpi_task->Validation());
  } else {
    mpi_task->Validation();
  }
}

TEST(NikitinaVMaxElemMatrValidation, FailsOnBothNegativeDims) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    auto *seq_task = new MaxElementMatrSEQ({-5, -1});
    ASSERT_FALSE(seq_task->Validation());
  }
  auto *mpi_task = new MaxElementMatrMPI({-5, -1});
  if (rank == 0) {
    ASSERT_FALSE(mpi_task->Validation());
  } else {
    mpi_task->Validation();
  }
}

TEST(NikitinaVMaxElemMatrValidation, FailsOnSizeMismatchTooFew) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    auto *seq_task = new MaxElementMatrSEQ({2, 2, 1, 2, 3});
    ASSERT_FALSE(seq_task->Validation());
  }
  auto *mpi_task = new MaxElementMatrMPI({2, 2, 1, 2, 3});
  if (rank == 0) {
    ASSERT_FALSE(mpi_task->Validation());
  } else {
    mpi_task->Validation();
  }
}

TEST(NikitinaVMaxElemMatrValidation, FailsOnSizeMismatchTooMany) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    auto *seq_task = new MaxElementMatrSEQ({2, 2, 1, 2, 3, 4, 5});
    ASSERT_FALSE(seq_task->Validation());
  }
  auto *mpi_task = new MaxElementMatrMPI({2, 2, 1, 2, 3, 4, 5});
  if (rank == 0) {
    ASSERT_FALSE(mpi_task->Validation());
  } else {
    mpi_task->Validation();
  }
}

}  // namespace nikitina_v_max_elem_matr
