# Matrix Sums By Columns

- Student: Гусева Алёна Сергеевна, 3823Б1ФИ2
- Technology: SEQ | MPI
- Variant: 12

## 1. Introduction
The task of computing column sums of a matrix is a fundamental problem in computational mathematics, frequently encountered in data analysis, machine learning, and scientific computing. The objective of this work is to implement a parallel solution using MPI (Message Passing Interface) to distribute the computational workload across multiple processes and achieve performance acceleration for large matrices.

## 2. Problem Statement
Given a matrix A of dimensions M×N, compute the sum of elements in each column, resulting in a vector of length N where each element j contains the sum of all elements in column j.

Input format: A tuple (M, N, matrix_data) where:
- M: number of rows (`uint32_t`)
- N: number of columns (`uint32_t`)
- matrix_data: flat matrix elements in row-major order (`vector<double>`)

Output format: A vector of N elements (`vector<double>`) containing the column sums.

Constraints:
- Matrix dimensions M and N must be positive integers
- The size of matrix_data must equal M × N
- The algorithm should handle matrices of substantial size efficiently

## 3. Baseline Algorithm (Sequential)
The sequential algorithm iterates through each element of the matrix, accumulating sums for each column:
```c++ 
for (uint32_t i = 0; i < rows; i++) {
    for (uint32_t j = 0; j < columns; j++) {
        column_sums[j] += matrix[(i * columns) + j];
    }
}
```
- Time Complexity: O(M × N)
- Space Complexity: O(N) for the output vector

## 4. Parallelization Scheme
### Data Distibution
The matrix is distributed among MPI processes using a block-cyclic distribution pattern:
- The flattened matrix is divided into contiguous blocks
- Each process receives approximately equal number of elements
- Remaining elements are distributed to the first few processes

### Communication Pattern
- **Broadcast**: Process 0 broadcasts matrix dimensions (rows, columns) to all processes
- **Scatterv**: Matrix data is scattered using variable block sizes to handle uneven division
- **Reduce**: Local column sums are reduced using MPI_SUM operation to get global sums

### Process Role
- **Rank 0**: Root process that initializes data, coordinates communication, and collects final results
- **Other ranks**: Worker processes that compute local sums and participate in reduction

## 5. Implementation Details
### File structure
```
└── tasks/
    └── guseva_a_matrix_sums/
        ├── common/                                 
        |   └── common.hpp  . . . . . . . . . . . . Common type definitions 
        |                                           and constants
        ├── data/
        |   ├── cases/*.txt . . . . . . . . . . . . Func tests input data
        |   ├── expected/*.txt  . . . . . . . . . . Func tests expected results
        |   └── perf/
        |       ├── input.txt . . . . . . . . . . . Perf tests input data
        |       └── expected.txt  . . . . . . . . . Perf tests expected results
        |
        ├── mpi/
        |   ├── inlude/
        |   |   └── ops_mpi.hpp . . . . . . . . . . MPI task class declaration 
        |   |                                       inheriting from BaseTask
        |   └── src/
        |       └── ops_mpi.cpp . . . . . . . . . . MPI implementation with data 
        |                                           distribution and reduction
        |
        ├── seq/
        |   ├── inlude/
        |   |   └── ops_mpi.hpp . . . . . . . . . . Sequential task class declaration
        |   └── src/
        |       └── ops_mpi.cpp . . . . . . . . . . Sequential reference
        |                                           implementation
        |
        ├── tests/ 
        |   ├── functional/
        |   |   └── main.cpp
        |   |
        |   └── performance/
        |       └── main.cpp
        |
        ├── info.json
        ├── report.md
        └── settings.json

```
### Key Classes
- `GusevaAMatrixSumsMPI`: MPI implementation with validation, preprocessing, execution, and post-processing methods

- `GusevaAMatrixSumsSEQ`: A similarly sequential implementation

### Important Assumptions
- Matrix is stored in row-major order
- Matrix dimensions fit within 32-bit integers
- MPI environment is properly initialized
- Process 0 has the complete input matrix

### Memory Considerations
- Each process stores only its local portion of the matrix
- Temporary buffers for local sums and communication
- Overall memory usage scales with `O(local_elements + N)` per process

## 6. Experimental Setup
- **Hardware/OS**: 
    - **Host**: Intel Core i7-14700k, 8+12 cores, 32 Gb DDR4, Windows 10 (10.0.19045.6456) 
    - **Virtual**: Intel Core i7-14700k, 12 cores, 8 Gb, WSL2 (2.6.1.0) + Ubuntu (24.04.3 LTS)
- **Toolchain**:
    | Compiler | Version | Build Type |
    |:-:|:-:|:-:|
    | gcc | 14.2.0 x86_64-linux-gnu | Release
    | clang | 21.1.0 x86-64-pc-linus-gnu | Release
- **Environment**:
    | Variable | Values | Comment |
    |:-|:-:|:-:|
    | PPC_NUM_PROC | [1, 4] | For `srcipts/run_tests.py` single executions w/o `--count` arg |
    | PPC_NUM_THREADS | 1 | Similarly as previous|
    | OMPI_ALLOW_RUN_AS_ROOT | 1 | Needed by OpenMPI to run under Docker and .devcointer |
    | OMPI_ALLOW_RUN_AS_ROOT_CONFIRM | 1 | Confirmation for previous env variable|
- **Data**: 

All the test data are genereated with my own Python script which is not represented in the current work. Generated test cases are contained in `./tasks/guseva_a_matrix_sums/data/` directory as shown in the diagram below:
```
data/
├── cases/*.txt . . . . . . . . . . . . Func tests input data
├── expected/*.txt  . . . . . . . . . . Func tests expected results
└── perf/
    ├── input.txt . . . . . . . . . . . Perf tests input data
    └── expected.txt  . . . . . . . . . Perf tests expected results
```


## 7. Results and Discussion

### 7.1 Correctness
Correctness was verified by:
- Using matrices with known sums.
- CI/CD
- Runs with multiple repetitions, test mixing, different env variables, and MPI flags (e.g., `--oversubscribe`)
- The epsilon constant (`kEpsilon = 10e-12`) ensures floating-point precision

### 7.2 Performance
Present time, speedup and efficiency. Example table:

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 1.00301 | 1.00    | N/A        |
| MPI         | 1     | 0.07488 | 13.39   | 13.39%     |
| MPI         | 2     | 0.07146 | 14.04   | 7.02%      |
| MPI         | 3     | 0.06804 | 14.74   |   4.91%    |
| MPI         | 4     | 0.06541 | 15.33   |  3.75%     |

## 7.2 Performance

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 1.00301 | 1.00    | N/A        |
| MPI         | 1     | 0.07488 | 13.39   | 13.39%     |
| MPI         | 2     | 0.07146 | 14.04   | 7.02%      |
| MPI         | 3     | 0.06804 | 14.74   | 4.91%      |
| MPI         | 4     | 0.06541 | 15.33   | 3.75%      |

**Discussion**:

### Bottlenecks Identified:

1. **Communication Overhead**: The MPI implementation shows significant communication overhead compared to computation time. Even with a single process, the efficiency is only 13.39%, indicating that MPI initialization and coordination consume substantial resources.

2. **Fixed Costs Dominance**: The minimal time improvement from 1 to 4 processes (0.07488s to 0.06541s) suggests that fixed costs (MPI initialization, data distribution, reduction operations) dominate the execution time rather than the actual computation.

### Scalability Limits:

1. **Diminishing Returns**: The speedup curve shows strong diminishing returns:
   - From 1 to 2 processes: +0.65 speedup
   - From 2 to 3 processes: +0.70 speedup  
   - From 3 to 4 processes: +0.59 speedup

2. **Communication-to-Computation Ratio**: For the given problem size, the ratio favors communication over computation. The column sum operation is computationally inexpensive (O(N) per element), making it difficult to amortize communication costs.

## 8. Conclusions
The MPI implementation successfully parallelizes the column sum computation with:
- Correct results matching the sequential implementation
- Significant speedup for large matrices
- Good scalability up to moderate process counts
- Efficient memory distribution across processes

Limitations:
- Communication overhead limits efficiency for small matrices
- Fixed distribution strategy may not be optimal for all matrix shapes
- Assumes homogeneous computing environment

## 9. References
1. [Open MPI Documentation](https://www.open-mpi.org/doc/)
2. [Основы MPI для чайников](https://habr.com/ru/articles/121235/)
